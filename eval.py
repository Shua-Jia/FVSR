import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils import data
from utils.models import Encoder, Decoder
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from pathlib import Path
from utils.tools import NoisePredictor,LatentDDIMSampler

def downgrade_in_seq(image, target_size, domain_list):
    ori_size = domain_list[0]
    # image = F.interpolate(image, size=domain_list[target_size], mode='bicubic', antialias=True)
    # image = F.interpolate(image, size=ori_size, mode='bicubic', antialias=True)

    for i in range(int(target_size)): # target_size = t
        image = F.interpolate(image, size=domain_list[i+1], mode='bicubic', antialias=True)
    image = F.interpolate(image, size=ori_size, mode='bicubic', antialias=True)

    ## Anti Alpha channel in .png format
    # if image.shape[1] == 4:
    #     image = image[:,:3,:,:]
    return image


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        parent_folder = path.parent
        # 再获取这个父级文件夹相对于 'input' 文件夹的路径
        target_path = parent_folder.relative_to('input')
        # 转成字符串（可选，默认是Path对象）
        target_path_str = str(target_path)
        filename=path.stem

        return self.transform(img),target_path_str,filename

transform = transforms.Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def inference(args, domains_interval):
    data_root = args.input_folder

    batch_size = 1
    im_size = 1024
    dataloader_workers = args.workers
    saved_image_folder = "result"
    if not os.path.exists(saved_image_folder):
        os.makedirs(saved_image_folder)

    num_domains = len(domains_interval)

    start = args.start
    target = args.target
    for i, domain in enumerate(domains_interval):
        if start >= domain:
            start_domain = i + 1
            break

    for i, domain in enumerate(reversed(domains_interval)):
        if target <= domain:
            break_domain = num_domains - i
            break

    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = torch.device("cpu")

    dataset = Dataset(data_root, im_size)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=dataloader_workers, pin_memory=False))

    encoder = Encoder()
    decoder = Decoder()

    if args.encoder_ckpt != 'None':
        print(f"Loading encoder from {args.encoder_ckpt}")
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))
        print("Encoder loaded.")

    if args.decoder_ckpt != 'None':
        print(f"Loading decoder from {args.decoder_ckpt}")
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
        print("Decoder loaded.")
    encoder.to(device)
    decoder.to(device)

    noise_predictor = NoisePredictor(c=1024).to(device)  
    if args.noise_predictor_ckpt != "None":
        print(f"Loading noise predictor from {args.noise_predictor_ckpt}")
        noise_predictor.load_state_dict(torch.load(args.noise_predictor_ckpt, map_location=device))
        noise_predictor.eval()
    else:
        raise ValueError("Noise predictor checkpoint must be provided!")


    ddim_sampler = LatentDDIMSampler(
        noise_predictor=noise_predictor,
        num_timesteps=args.num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )

    for n in tqdm(range(len(dataset))):

        gt_img,dir,filename = next(dataloader)
        gt_img=gt_img.to(device)
        dir=dir[0]
        filename=filename[0]
        lr_img = downgrade_in_seq(gt_img.clone(), start_domain-1, domains_interval)

        with torch.no_grad():
            sr_img = lr_img.clone()
            previous_prediction = None
            inverse_momentum = 0
            for i in reversed(range(1, start_domain)):
                t = torch.ones(1, device=device).long() * i
                if previous_prediction is None:
                    inverse_momentum_LR = 0
                else:
                    inverse_momentum_LR = downgrade_in_seq(inverse_momentum.clone(), t, domains_interval)

                l, h = encoder(sr_img + inverse_momentum_LR, t)

                # denoised_latent = ddim_sampler.ddim_sample_loop(
                #     latent_img_lr=l,
                #     num_ddim_steps=args.ddim_steps,
                #     eta=args.eta
                # )
                # denoised_latent = (denoised_latent - denoised_latent.min()) / (denoised_latent.max() - denoised_latent.min())
                import matplotlib.pyplot as plt
                import numpy as np
                #
              
                # latent_vec =l*0.5+ denoised_latent*0.5  
                # feature_maps = latent_vec.squeeze(0)  

                # feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

     
                # grid_size = 32
                # grid = np.zeros((grid_size * 4, grid_size * 4))+0.2
                # for i in range(grid_size):
                #     for j in range(grid_size):
                #         idx = i * grid_size + j
                #         grid[i * 4: (i + 1) * 4, j * 4: (j + 1) * 4] = feature_maps[idx].numpy()
                #
              
                # plt.figure(figsize=(10, 10))
                # plt.imshow(grid, cmap='gray')  # 用灰度图显示，也可换其他colormap
                # plt.axis('off')
                # plt.show()


               # result = decoder(denoised_latent*0+l, t, h)
                result = decoder(l, t, h)
                temp_result = result.clone()
                if previous_prediction is None:
                    previous_prediction = temp_result.clone()
                inverse_momentum += (previous_prediction - temp_result)
                previous_prediction = temp_result.clone()

                if i == break_domain:
                    break
                else:
                    sr_img = downgrade_in_seq(result, t - 1, domains_interval)
        save_img = F.interpolate(result, target, mode='bicubic', antialias=True)

 
        save_path = os.path.join(saved_image_folder, dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        vutils.save_image(save_img.add(1).mul(0.5), os.path.join(save_path, f'{filename}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FVSR')
    parser.add_argument('--input_folder', type=str, default='input',
                        help='path of resource testing images, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--start', type=int, default=256, help='start of resolution')
    parser.add_argument('--target', type=int, default=1024, help='Target of super-resolution')
    # parser.add_argument('--ckpt', type=str, default="train_results/350000.pth")
    parser.add_argument('--encoder_ckpt', type=str,default=r'./train_results/64_1024_Universal/models/encoder.pth')
    parser.add_argument('--decoder_ckpt', type=str,default=r'./train_results/64_1024_Universal/models/decoder.pth')
    parser.add_argument('--noise_predictor_ckpt', type=str,
                        default=r'.\train_results\64_1024_Universal\models\noise_predictor.pth')
 
    parser.add_argument('--num_timesteps', type=int, default=2000, help='Total diffusion timesteps (same as training)')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps (smaller=faster)')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0=deterministic, 1=stochastic)')

    parser.add_argument('--cuda', type=int, default=0, help='GPU index to use')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')


    domains_interval = [1024, 768, 512, 324, 256, 128, 64]

    args = parser.parse_args()
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    inference(args, domains_interval)
