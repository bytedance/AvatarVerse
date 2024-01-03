# from .sd import StableDiffusion
import torch
# import torch.nn.functional as F


from xml.etree.ElementTree import tostring


def disable_params_grad(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def prepare_text_embeddings(diffusion_net, text, negative='', dir_text=True, suppress_face=True):
    if text is None:
        print(f"[WARN] text prompt is not provided.")
        return None

    if not dir_text:
        return torch.stack([diffusion_net.get_text_embeds([text], [negative])], dim=0)
    else:
        text_zs = []
        view_txts = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
        for d in view_txts:
            # construct dir-encoded text
            text_cur = f"{text}, {d} view, {d} view"
            negative_text = f"{negative}"
            print(text_cur)
            # explicit negative dir-encoded text
            # if suppress_face:
            #     if negative_text != '': negative_text += ', '
            #     if d == 'back':
            #         negative_text += "face, front view"
            #     # elif d == 'front': negative_text += ""
            #     elif d == 'side':
            #         negative_text += "face, front view"
            #     elif d == 'overhead':
            #         negative_text += "face, front view"
            #     elif d == 'bottom':
            #         negative_text += "face, front view"
            # torch.cuda.empty_cache()
            text_zs.append(diffusion_net.get_text_embeds([text_cur], [negative_text]))
        diffusion_net.release() # release first, avoid oom error on v100
        # text_zs = torch.cat(text_zs, dim=0)
        text_zs = torch.stack(text_zs, dim=0)
        # print(text_zs.shape)
        # print(text_zs[:, 0, :5, 0])
        return text_zs


def guidance_for_views(max_guidance, dirs):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    guidance_scale_tensor = torch.ones_like(dirs)
    guidance_scale_tensor[dirs==0] = max_guidance
    
    guidance_scale_tensor[dirs==1] = max_guidance/10.
    guidance_scale_tensor[dirs==3] = max_guidance/10
    guidance_scale_tensor[dirs==4] = max_guidance/10.

    guidance_scale_tensor[dirs==2] = max_guidance/20.
    guidance_scale_tensor[dirs==5] = max_guidance/20.
    return guidance_scale_tensor