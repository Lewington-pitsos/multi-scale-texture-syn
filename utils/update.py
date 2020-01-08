from torch import optim
from utils import pyramid, img, stats
import torch.nn.functional as F

class Transferrer():
    def __init__(self, params):
        self.saver = img.Saver(params["save_mode"])
        self.style_feature_pyr = params["style_feature_pyr"]
        self.target_features = params["content_features"]
        self.save_mode = params["save_mode"]
        self.reset_save()
    
    def reset_save(self):
        self.save_at = 1 if self.save_mode == "throughout" else 0
    
    def transfer(self, params):
        if self.save_at == 0:
            self.save_at = params["iterations"]
        
        stepper_params = params.copy()
        opt_img = params["image"]
        optimizer = optim.LBFGS([opt_img], lr=params["lr"])
        
        stepper_params["optimizer"] = optimizer
        stepper_params["opt_img"] = opt_img
        stepper_params["style_feature_pyr"] = self.style_feature_pyr
        stepper_params["target_features"] = self.target_features
        
        stepper = GradientStepper(stepper_params)
        
        self.saver.prepair_save_directory(params)

        for i in range (params["iterations"]):
#             if opt_img.grad is not None:
#                 print(opt_img.grad[0][0][0][:4])

            optimizer.step(stepper.closure)

            # occationally save an image so see how generation is going
            if (i + 1) == self.save_at:
                self.saver.save_image(opt_img, self.save_at)
                self.save_at *= 2
        
        self.reset_save()
        return opt_img, stepper.losses


class GradientStepper():
    gaussian_kernel = pyramid.build_gauss_kernel(n_channels=3, cuda=True)
    
    def __init__(self, params):
        self.optimizer = params["optimizer"]
        self.sample = params["opt_img"]
        self.style_feature_pyr = params["style_feature_pyr"]
        self.model = params["model"]
        self.style_hooks = params["style_hooks"]
        self.style_scale = params["style_scale"]
        self.losses = []
        self.max_levels = params["max_levels"]
        
        self.content_hook = params["content_hook"]
        self.target_features = params["target_features"]

    def style_loss(self, sample_feature_pyr):
        loss = 0
        
        for i in range(len(sample_feature_pyr)):
            sample_stats = sample_feature_pyr[i]
            target_stats = self.style_feature_pyr[i]
            
            for j in range(len(sample_stats)):
                loss += F.mse_loss(sample_stats[j], target_stats[j])

        return loss * self.style_scale
    
    
    def content_loss(self, content_features):
        loss = F.mse_loss(content_features, self.target_features)
        return loss * 100
    
    def loss_fn(self, sample_feature_pyr, content_features):
        return self.style_loss(sample_feature_pyr) + self.content_loss(content_features)
    
    def gaussian_pyramid(self):
        return pyramid.gaussian_pyramid(self.sample, self.gaussian_kernel, max_levels=self.max_levels)
    
    def closure(self):
        self.optimizer.zero_grad() # Please read up on this if you don't know what it does. 
        
         # sample content features
        sample_content = stats.extract_features(self.model, [self.content_hook], self.sample)[0]
        
        # sample style features
        sample_pyr = self.gaussian_pyramid()
        sample_feature_pyr = []
        
        for sample in sample_pyr:
            sample_texture_stats = stats.extract_features(self.model, self.style_hooks, sample, stats.gram_matrix)
            sample_feature_pyr.append(sample_texture_stats)
       
        loss = self.loss_fn(sample_feature_pyr, sample_content)
        self.losses.append(loss)
        loss.backward()

        return loss