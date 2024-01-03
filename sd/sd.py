from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

# suppress partial model loading warning
# logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def dynamic_threshold(x, percent=0.95):
    #return x
    x_ = x.view(x.size(0), -1).abs()
    s = torch.quantile(x_, percent, dim=-1)
    s.clamp_(min = 1.)
    s = s[:, None, None, None]
    x = x.clamp(-s, s) / s
    return x

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='1.5', random_sample=True, n_iters=None, time_range=(0.02, 0.98)):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(
                f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * time_range[0])
        self.max_step = int(self.num_train_timesteps * time_range[1])

        print(f'[INFO] loading stable diffusion...')
        try:
            if sd_version == '1.5':

                model_key = 'runwayml/stable-diffusion-v1-5'
                self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
                self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
                # 4. Create a scheduler for inference
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                            num_train_timesteps=self.num_train_timesteps)

            elif sd_version == '2.1':
                print('training with stable diffusion 2.1')
                # model_key = '/root/code/DirectVoxGO/stable-diffusion-2-1-base'
                model_key = 'stabilityai/stable-diffusion-2-1-base'
                self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
                self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
                
                # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                            num_train_timesteps=self.num_train_timesteps)
        except Exception as e:
            raise e
            model_key = "runwayml/stable-diffusion-v1-5"

            # Create model
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

            # self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler")
            self.scheduler = PNDMScheduler.from_config(model_key, subfolder="scheduler")
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.random_sample = random_sample
        self.n_iters = n_iters
        if not self.random_sample:
            assert isinstance(n_iters, int)
            self.scheduler.set_timesteps(n_iters)
            self.n_iters = len(self.scheduler.timesteps)
        print(f'[INFO] loaded stable diffusion!')

    def release(self):
        del self.text_encoder
        del self.tokenizer
        torch.cuda.empty_cache()

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, latent_img=False, n_sample=1, accelerator=None, **kwargs):
        
        # interp to 512x512 to be fed into vae.
        guidance_scale_tensor = guidance_scale.reshape([-1, 1, 1, 1])
        if len(guidance_scale_tensor) != len(pred_rgb):
            guidance_scale_tensor = torch.cat([guidance_scale_tensor] * len(pred_rgb), dim=0)
        text_embeddings = text_embeddings
        if latent_img:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512.to(self.device))
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        if n_sample > 1:
            # make n_sample copies for sampling differen t and noise
            text_embeddings = torch.cat([text_embeddings]*n_sample, 0)
            latents = torch.cat([latents]*n_sample, 0)
            guidance_scale_tensor = torch.cat([guidance_scale_tensor]*n_sample, 0)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [len(latents)], dtype=torch.long, device=self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            text_embeddings = torch.cat([text_embeddings[:, 0], text_embeddings[:, 1]], dim=0)
            # latents.data = latents.data.clip(-1, 1)
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_model_input = torch.cat([t] * 2)
            # print(latent_model_input.shape, t.shape, text_embeddings.shape)
            noise_pred = self.unet(latent_model_input, t_model_input, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale_tensor * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).reshape([-1, 1, 1, 1])
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad)
        # latents.backward(gradient=grad, retain_graph=True)
        if accelerator is None:
            latents.backward(gradient=grad, retain_graph=True)
        else:
            accelerator.backward(latents, gradient=grad, retain_graph=True)
        # # return torch.mean(grad)  # dummy loss value
        return torch.mean(torch.abs(grad))  # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='Peter Rabbit wearing a blue denim shirt')
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, sd_version='1.5')
    for g in [2,3,4,5,6,7,8,9,10,20,30,40,50,60,80,100]:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, guidance_scale=g)

        # visualize image
    
        plt.imshow(imgs[0])
        plt.savefig('./img_%d.png'%g)

