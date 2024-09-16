import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm 



def train_generator(generator_G:torch.nn.Module,
                    generator_H:torch.nn.Module,
                    discriminator_X:torch.nn.Module,
                    discriminator_Y:torch.nn.Module,
                    batch:tuple,
                    mse_loss:torch.nn.Module,
                    l1_loss:torch.nn.Module,
                    cycle_lambda:float,
                    optimizer:torch.optim.Optimizer,
                    scaler:torch.cuda.amp.GradScaler,
                    device:torch.device):
    
    real_apple, real_orange = batch['Apple'].to(device), batch['Orange'].to(device)
    
    generator_G.train()
    generator_H.train()
    discriminator_X.eval()
    discriminator_Y.eval()
    
    with torch.cuda.amp.autocast(enabled=False):
        
        # Generating fakes
        fake_orange = generator_G(real_apple)
        fake_apple = generator_H(real_orange)
        
        # Adversarial Loss
        fake_apple_preds = discriminator_X(fake_apple)
        fake_apple_labels = torch.ones_like(fake_apple_preds)
        adversarial_loss_1 = mse_loss(fake_apple_preds, fake_apple_labels)
        fake_orange_preds = discriminator_Y(fake_orange)
        fake_orange_labels = torch.ones_like(fake_orange_preds)
        adversarial_loss_2 = mse_loss(fake_orange_preds, fake_orange_labels)
        adversarial_loss = adversarial_loss_1 + adversarial_loss_2
        
        # Cycle Loss
        cycle_orange = generator_G(fake_apple)
        cycle_apple = generator_H(fake_orange)
        cycle_loss_1 = l1_loss(cycle_apple, real_apple)
        cycle_loss_2 = l1_loss(cycle_orange, real_orange)
        cycle_loss = (cycle_loss_1 + cycle_loss_2) * cycle_lambda

        # Total loss
        loss = adversarial_loss + cycle_loss
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del real_apple, real_orange, fake_apple, fake_orange, fake_apple_labels, fake_orange_labels, fake_apple_preds, fake_orange_preds, 
    cycle_apple, cycle_orange, adversarial_loss_1, adversarial_loss_2, adversarial_loss, cycle_loss_1, cycle_loss_2, cycle_loss

    return loss.item()



def train_discriminator(generator_G:torch.nn.Module,
                        generator_H:torch.nn.Module,
                        discriminator_X:torch.nn.Module,
                        discriminator_Y:torch.nn.Module,
                        batch:tuple,
                        mse_loss:torch.nn.Module,
                        optimizer:torch.optim.Optimizer,
                        scaler:torch.cuda.amp.GradScaler,
                        device:torch.device):
    
    real_apple, real_orange = batch['Apple'].to(device), batch['Orange'].to(device)
    
    generator_G.eval()
    generator_H.eval()
    discriminator_X.train()
    discriminator_Y.train()
    
    with torch.cuda.amp.autocast(enabled=False):
        
        # Generating fakes
        fake_orange = generator_G(real_apple)
        fake_apple = generator_H(real_orange)
        
        # Loss for discriminator X
        real_apple_preds = discriminator_X(real_apple)
        fake_apple_preds = discriminator_X(fake_apple)
        real_apple_labels = torch.ones_like(real_apple_preds)
        fake_apple_labels = torch.zeros_like(fake_apple_preds)
        apple_loss_1 = mse_loss(real_apple_preds, real_apple_labels)
        apple_loss_2 = mse_loss(fake_apple_preds, fake_apple_labels)
        apple_loss = apple_loss_1 + apple_loss_2
        
        # Loss for discriminator Y
        real_orange_preds = discriminator_Y(real_orange)
        fake_orange_preds = discriminator_Y(fake_orange)
        real_orange_labels = torch.ones_like(real_orange_preds)
        fake_orange_labels = torch.zeros_like(fake_orange_preds)
        orange_loss_1 = mse_loss(real_orange_preds, real_orange_labels)
        orange_loss_2 = mse_loss(fake_orange_preds, fake_orange_labels)
        orange_loss = orange_loss_1 + orange_loss_2
        
        # Total loss
        loss = apple_loss + orange_loss
        
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del real_apple, real_orange, fake_apple, fake_orange, real_apple_labels, fake_apple_labels, real_apple_preds, 
    fake_apple_preds, real_orange_labels,fake_orange_labels, real_orange_preds, fake_orange_preds, 
    apple_loss_1, apple_loss_2, apple_loss, orange_loss_1, orange_loss_2, orange_loss
    
    return loss.item()



def train_models(generator_G:torch.nn.Module,
                 generator_H:torch.nn.Module,
                 discriminator_X:torch.nn.Module,
                 discriminator_Y:torch.nn.Module,
                 dataloader:torch.utils.data.DataLoader,
                 mse_loss:torch.nn.Module,
                 l1_loss:torch.nn.Module,
                 cycle_lambda:float,
                 gen_optimizer:torch.optim.Optimizer,
                 disc_optimizer:torch.optim.Optimizer,
                 gen_scaler:torch.cuda.amp.GradScaler,
                 disc_scaler:torch.cuda.amp.GradScaler,
                 device:torch.device,
                 NUM_EPOCHS:int,
                 generator_G_path:str=None,
                 generator_H_path:str=None,
                 discriminator_X_path:str=None,
                 discriminator_Y_path:str=None,
                 result_path:str=None):
    
    
    for epoch in range(24, NUM_EPOCHS+1):
        
        gen_loss = 0
        disc_loss = 0
        
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            
            for i, batch in t:
                
                disc_batch_loss = train_discriminator(generator_G=generator_G, 
                                                      generator_H=generator_H, 
                                                      discriminator_X=discriminator_X, 
                                                      discriminator_Y=discriminator_Y, 
                                                      batch=batch, 
                                                      mse_loss=mse_loss, 
                                                      optimizer=disc_optimizer, 
                                                      scaler=disc_scaler, 
                                                      device=device, )
                
                gen_batch_loss = train_generator(generator_G=generator_G,
                                                 generator_H=generator_H, 
                                                 discriminator_X=discriminator_X, 
                                                 discriminator_Y=discriminator_Y,
                                                 batch=batch,
                                                 mse_loss=mse_loss,
                                                 l1_loss=l1_loss,
                                                 cycle_lambda=cycle_lambda,
                                                 optimizer=gen_optimizer,
                                                 scaler=gen_scaler,
                                                 device=device)
                
                gen_loss += gen_batch_loss
                disc_loss += disc_batch_loss
                
                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}]')
                t.set_postfix({
                    'Gen batch loss' : gen_batch_loss,
                    'Gen train loss' : gen_loss/(i+1),
                    'Disc batch loss' : disc_batch_loss,
                    'Disc train loss' : disc_loss/(i+1)
                })
                
                if generator_G_path and generator_H_path and discriminator_X_path and discriminator_Y_path:
                    torch.save(obj=generator_G.state_dict(), f=generator_G_path)
                    torch.save(obj=generator_H.state_dict(), f=generator_H_path)
                    torch.save(obj=discriminator_X.state_dict(), f=discriminator_X_path)
                    torch.save(obj=discriminator_Y.state_dict(), f=discriminator_Y_path)       
                    
                if i % 100 == 0 and result_path:
                    
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}_crct_plot.png'

                    generator_G.eval()
                    generator_H.eval()
                    with torch.inference_mode():
                        real_apple, real_orange = batch['Apple'].to(device), batch['Orange'].to(device)       
                        fake_orange = generator_G(real_apple)
                        fake_apple = generator_H(real_orange)
                        cycle_orange = generator_G(fake_apple)
                        cycle_apple = generator_H(fake_orange)
                
                
                    fin_tensor = torch.cat([real_apple, fake_orange, cycle_apple, real_orange, fake_apple, cycle_orange], dim=0)
                    grid = make_grid(fin_tensor, nrow=3, normalize=True, padding=16, pad_value=1)
                    fig = plt.figure()
                    plt.imshow(grid.permute(1,2,0).cpu())
                    plt.title(f'Epoch {epoch} set {(i//100)+1}', fontweight='bold', fontsize=16)
                    plt.axis(False);
                    
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)
                    
                    
                    