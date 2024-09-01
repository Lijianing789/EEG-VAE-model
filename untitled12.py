### adding loss curve 
##  changing the x-time(s)
#   adding validation dataset




#%%   VAE model
######下一步改 filtering !!!!!!

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random
import scipy.stats
from scipy import signal
from scipy.signal import butter, filtfilt, detrend, decimate
# Model Hyperparameters
cuda = True
DEVICE = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

batch_size = 100 # Adjust batch size as needed

# Parameters for the model
num_channels = 1  # Each sample has one channel, as each segment is 800 data points of 1D
segment_length = 800  # Each segment has 4 seconds of data, with a sampling frequency of 200 Hz, so 800 time points
x_dim = num_channels * segment_length

hidden_dim1 = 800
hidden_dim2 = 400 # Adding a second hidden dimension for the decoder
latent_dim = 20
learning_rate= 0.000001
epochs =100
dropout_rate = 0.1  # Dropout rate for regularization

def zerocentered(data):
    data_mean=data.mean(dim=1, keepdim=True)
    return data-data_mean


def normalization(data):
    data_min = data.min(dim=-1, keepdim=True)[0]
    data_max = data.max(dim=-1, keepdim=True)[0]
    normalization_data =(data - data_min) / (data_max - data_min+1e-6)
    return normalization_data

# Custom Dataset
class EEGDataset(Dataset):
    def __init__(self, noisy_data, clean_data, segment_length, overlap=0.5):
        self.clean_data = torch.tensor(clean_data, dtype=torch.float32)
        self.noisy_data = torch.tensor(noisy_data, dtype=torch.float32)
        self.segment_length = segment_length
        self.overlap = overlap
        self.clean_segments = self.segment_data(self.clean_data, segment_length, overlap)
        self.noisy_segments = self.segment_data(self.noisy_data, segment_length, overlap)
        
    def segment_data(self, data, segment_length, overlap):
        step = int(segment_length * (1 - overlap))  # Calculate step length
        num_segments = (data.shape[1] - segment_length) // step + 1  # Calculate total number of segments
        segments = []
        for i in range(num_segments):
            start = i * step
            end = start + segment_length
            segment = data[:, start:end]
            segment = normalization(segment) #normalize
            segment = zerocentered(segment)
           
         
            segments.append(segment)
        segments = torch.cat(segments, dim=0)
        return segments.view(-1, num_channels * segment_length)
    
    def __len__(self):
        return len(self.noisy_segments)
    
    def __getitem__(self, idx):
        return self.noisy_segments[idx], self.clean_segments[idx], 


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim1)
        self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var = nn.Linear(hidden_dim2, latent_dim)
        self.ReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.training = True

    def forward(self, x):
        h_ = self.ReLU(self.FC_input(x))
        h_ = self.dropout(h_)
        h_ = self.ReLU(self.FC_hidden(h_))
        h_ = self.dropout(h_)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim2, hidden_dim1, output_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim2)
        self.FC_hidden2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.FC_output = nn.Linear(hidden_dim1, output_dim)
        
        self.ReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optional: Add Batch Normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim1)
        
    def forward(self, x):
        h = self.ReLU(self.FC_hidden(x))
        h = self.batch_norm1(h)  # Apply batch normalization
        h = self.dropout(h)
        h = self.ReLU(self.FC_hidden2(h))
        h = self.batch_norm2(h)  # Apply batch normalization
        h = self.dropout(h)
        x_regenereted = torch.sigmoid(self.FC_output(h))
        return x_regenereted

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_regenereted = self.Decoder(z)
        return x_regenereted, mean, log_var




def loss_function(x, x_regenereted, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_regenereted, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD



# def loss_function(x, x_regenereted, mean, log_var,beta=0.5):
#     reproduction_loss = nn.functional.mse_loss(x_regenereted, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     return reproduction_loss + beta*KLD


#%%
# def lowpass_filter(data, cutoff=0.2, fs=1.0, order=2):

#     # 计算归一化截止频率
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
    
#     # 设计Butterworth低通滤波器
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
#     # 对每个样本应用滤波器
#     filtered_data = []
#     for i in range(data.size(0)):
#         sample = data[i].detach().numpy()  # 将张量转换为NumPy数组
#         filtered_sample = filtfilt(b, a, sample)  # 应用滤波器
#         filtered_data.append(filtered_sample)
#     return torch.tensor(filtered_data, dtype=torch.float32).to(data.device)


# def filter_coefficients(cutoff, fs, order, ftype):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype=ftype, analog=False)
#     return b, a

# # Filter data using filtfilt command
# def signal_filter(data, cutoff, fs, order, ftype):
#     b, a = filter_coefficients(cutoff, fs, order, ftype)
#     signal_filtered = filtfilt(b, a, data)
#     return signal_filtered

# def highpass_filtering(data, cutoff, fs, order, ftype):
#     # Get nyquist frequency of signal
#     nyq = 0.5 * fs
#     # Find the normalised cut-off frequency
#     cutoff = 1/nyq  
#     # Generate array of filter co-efficients
#     b, a = signal.butter(order, cutoff, btype=ftype, analog=False)
#     filtered_data = signal.filtfilt(b, a, data)
#     return filtered_data
def highpass_filtering(data, cutoff, fs, order):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    nyq = 0.5 * fs
    norm_cutoff = cutoff/ nyq  
    b, a = signal.butter(order, norm_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    filtered_data_copy = np.copy(filtered_data)
    filtered_data_tensor = torch.tensor(filtered_data_copy)
    return filtered_data_tensor


def bandpass_filtering(data, lowcut, highcut, fs, order):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    filtered_data_copy = np.copy(filtered_data)
    filtered_data_tensor = torch.tensor(filtered_data_copy)
    return filtered_data_tensor

#%%




def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    n = len(arr)
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
    #Calculate Mean
    mean = (square / (float)(n)+1e-10)
    #Calculate Root
    root = math.sqrt(mean)
    return root

def RRMSE(true, pred):

    num = rmsValue(true-pred)
    den = rmsValue(true)
    rrmse_loss = num/den
    return rrmse_loss


# calcualte RMSE (Root Mean Square Error) 
def RMSE(true, pred):
    return rmsValue(true-pred)

#%%
#metric
def calculate_metrics_mse(original, reconstructed):
    # Ensure original and reconstructed tensors are in the same range
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def calculate_metrics_RRMSEt(original, reconstructed):
    # Ensure original and reconstructed tensors are in the same range
    original = original.cpu()
    reconstructed = reconstructed.cpu()
    rmse = torch.sqrt(torch.mean((original - reconstructed) ** 2))    
    # Calculate standard deviation of the original data
    std_original = torch.std(original)    
    # Calculate RRMSE
    rrmse = rmse / (std_original + 1e-10)  # Add small epsilon to avoid division by zero  
    return rrmse.item()
 #%%   


def train_vae():
    print("Start training VAE...")
    model.to(DEVICE)
    model.train()
    epoch_reconstructed_train_data = []  # Store regenerated data from the final epoch
    epoch_reconstructed_val_data = [] 
    train_loss_list = []
    val_loss_list = []


    for epoch in range(epochs):
        mse_train_list = []
        RRMSEt_train_list = []
        train_loss = 0
        batch_reconstructed_train_data = []  # Reset at the start of each epoch
        for batch_idx, (noisy_segments, clean_segments) in enumerate(train_loader):
            clean_train_segment = clean_segments.to(DEVICE)
            noisy_train_segment = noisy_segments.to(DEVICE)
            optimizer.zero_grad()
            reconstructed_train_segment, mean, log_var = model(noisy_train_segment)
            #reconstructed_train_segment = lowpass_filter(reconstructed_train_segment)
            reconstructed_train_segment = normalization(reconstructed_train_segment)  # Normalize the regenerated data 
            reconstructed_train_segment = zerocentered(reconstructed_train_segment)
            reconstructed_train_segment = bandpass_filtering(reconstructed_train_segment, 1, 40, 200, 8)
            

            loss = loss_function(clean_train_segment, reconstructed_train_segment, mean, log_var)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_reconstructed_train_data.append(reconstructed_train_segment.cpu().detach())
            
        epoch_reconstructed_train_data = torch.cat(batch_reconstructed_train_data, dim=0)  # Shape: [train_size, 800]
        train_loss_list.append(train_loss/ train_dataset_size) 
        print(f"Epoch {epoch + 1} complete! \tAverage Loss: {train_loss_list[-1]:.4f}")

         
     
        #print(f"Batch {batch_idx+1}: clean_test_segment size = {clean_test_segment.size()}, reconstructed_test_segment size = {reconstructed_test_segment.size()}, mse = {mse}")

 #%%validation   
    
        model.eval()
        mse_val_list = []
        RRMSEt_val_list = []
        val_loss=0
        batch_reconstructed_val_data = []
        with torch.no_grad():
            for batch_idx, (noisy_segments, clean_segments) in enumerate(val_loader):
                clean_val_segment = clean_segments.to(DEVICE)
                noisy_val_segment = noisy_segments.to(DEVICE)
                reconstructed_val_segment, mean, log_var = model(noisy_val_segment)
                #reconstructed_val_segment = lowpass_filter(reconstructed_val_segment)
                reconstructed_val_segment = normalization(reconstructed_val_segment)  # Normalize the regenerated data 
                reconstructed_val_segment = zerocentered(reconstructed_val_segment)
                reconstructed_val_segment = bandpass_filtering(reconstructed_val_segment, 1, 40, 200, 8)

                
                loss = loss_function(clean_val_segment, reconstructed_val_segment, mean, log_var)
                val_loss += loss.item()
                batch_reconstructed_val_data.append(reconstructed_val_segment.cpu().detach())

             
            epoch_reconstructed_val_data = torch.cat(batch_reconstructed_val_data, dim=0)  # Shape: [train_size, 800]
        val_loss_list.append(val_loss / val_dataset_size)
        epoch_reconstructed_val_data = torch.cat(batch_reconstructed_val_data, dim=0)  # Shape: [train_size, 800]
        
  ### metric      
    for i in range(train_dataset_size):
       # print(len(train_dataset.clean_segments))
        segment_clean = train_dataset.clean_segments[i]
        segment_reconstructed = epoch_reconstructed_train_data[i]
       # print(len(epoch_reconstructed_train_data))
        mse = calculate_metrics_mse(segment_clean, segment_reconstructed)
        RRMSEt = RRMSE(segment_clean, segment_reconstructed)
        mse_train_list.append(mse)
        RRMSEt_train_list.append(RRMSEt)          

    mse_train_list = np.array(mse_train_list)
    RRMSEt_train_list = np.array(RRMSEt_train_list)  
       
    for i in range(val_dataset_size): 
        segment_clean = val_dataset.clean_segments[i]
        segment_reconstructed = epoch_reconstructed_val_data[i]
        mse = calculate_metrics_mse(segment_clean, segment_reconstructed)
        RRMSEt = RRMSE(segment_clean, segment_reconstructed)
        mse_val_list.append(mse)
        RRMSEt_val_list.append(RRMSEt)       
    mse_val_list = np.array(mse_val_list)
    RRMSEt_val_list = np.array(RRMSEt_val_list)                 
       
    return epoch_reconstructed_train_data,epoch_reconstructed_val_data, mse_train_list,RRMSEt_train_list, mse_val_list, RRMSEt_val_list, train_loss_list, val_loss_list


# def calculate_metrics_CC(original, reconstructed):
#     correlations = []
#     original = original = original.detach().cpu().numpy()
#     reconstructed = reconstructed.detach().cpu().numpy()
#     for i in range(original.shape[0]): 
#         correlation, _ = scipy.stats.pearsonr(original[i], reconstructed[i])
#         correlations.append(correlation)    
#     return correlations

def test_vae():
    
    model.eval()
    all_test_reconstructed_segments = []
    mse_list = []
    RRMSEt_list = []
   # CC = []
    with torch.no_grad():
        for batch_idx, (noisy_segments, clean_segments) in enumerate(tqdm(test_loader)):
            clean_test_segment = clean_segments.to(DEVICE)
            noisy_test_segment = noisy_segments.to(DEVICE)
            reconstructed_test_segment, _, _ = model(noisy_test_segment)
           # reconstructed_test_segment = replace_jumps_with_mean(reconstructed_test_segment, 0.3)
            #reconstructed_test_segment = lowpass_filter(reconstructed_test_segment)
            reconstructed_test_segment = normalization(reconstructed_test_segment) 
            reconstructed_test_segment = zerocentered(reconstructed_test_segment)
            reconstructed_test_segment = bandpass_filtering(reconstructed_test_segment, 1, 40, 200, 8)
             # Normalize the regenerated data         
  
            all_test_reconstructed_segments.append(reconstructed_test_segment.cpu().detach())
           # print("*clean test size", "%.4f" % clean_test_segment.size(0))
            for i in range(clean_test_segment.size(0)): 
                segment_clean = clean_test_segment[i]
                segment_reconstructed = reconstructed_test_segment[i]
                mse = calculate_metrics_mse(segment_clean, segment_reconstructed)
                RRMSEt = RRMSE(segment_clean, segment_reconstructed)
                mse_list.append(mse)
                RRMSEt_list.append(RRMSEt)
            
            #print(f"Batch {batch_idx+1}: clean_test_segment size = {clean_test_segment.size()}, reconstructed_test_segment size = {reconstructed_test_segment.size()}, mse = {mse}")
    all_test_reconstructed_segments = torch.cat(all_test_reconstructed_segments, dim=0)  # Shape: [test_size, 800]
    mse_list = np.array(mse_list)
    RRMSEt_list = np.array(RRMSEt_list)
    #CC = calculate_metrics_CC(clean_test_segment, reconstructed_test_segment)
    return all_test_reconstructed_segments, mse_list, RRMSEt_list



def show_images(original, reconstructed, unnormalized, idx, title, fs=256):
    plt.figure(figsize=(24, 6))  # 调整画布大小以容纳三张图
    
    # 动态生成时间轴以适应未归一化信号的长度
    unnormalized_length = unnormalized[idx].shape[-1]
    time_axis_unnormalized = np.linspace(0, unnormalized_length / fs, unnormalized_length, endpoint=False)
    
    # 动态生成时间轴以适应归一化信号的长度
    normalized_length = original[idx].shape[-1]
    time_axis_normalized = np.linspace(0, normalized_length / fs, normalized_length, endpoint=False)
    
    # 动态生成时间轴以适应重构信号的长度
    reconstructed_length = reconstructed[idx].shape[-1]
    time_axis_reconstructed = np.linspace(0, reconstructed_length / fs, reconstructed_length, endpoint=False)
    
    # Plot unnormalized original data in green
    plt.subplot(1, 3, 1)
    plt.plot(time_axis_unnormalized, unnormalized[idx].flatten(), color='#2ca02c')
    plt.title(f'Unnormalized {title} Data at segment {idx+1}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude (μV)')  # 显示未归一化信号的振幅
    
    # Plot original data (normalized) in blue
    plt.subplot(1, 3, 2)
    plt.plot(time_axis_normalized, original[idx].flatten(), color='#1f77b4')
    plt.title(f'Original {title} Data at segment {idx+1}', fontsize=14)
    plt.xlabel('Time(s)', fontsize=14)
    plt.ylabel('Normalized Amplitude', fontsize=14)
    
    # Plot reconstructed data in orange
    plt.subplot(1, 3, 3)
    plt.plot(time_axis_reconstructed, reconstructed[idx].flatten(), color='#ff7f0e')
    plt.title(f'Reconstructed {title} Data at segment {idx+1}', fontsize=14)
    plt.xlabel('Time(s)' ,fontsize=14)
    plt.ylabel('Normalized Amplitude' ,fontsize=14)
    
    plt.tight_layout()
    plt.show()




def show_image_constrast(original, reconstructed, idx, title):
    time_interval=4.0
    plt.figure(figsize=(18, 9))
    time_axis= np.linspace(0, time_interval, int(fs * time_interval), endpoint=False)

    line_width = 1.0
    for i, idx in enumerate(idx):
        plt.subplot(2, 4, i + 1)  
        plt.plot(time_axis, original[idx].flatten(), label='Clean EEG input', linewidth=line_width)
        plt.plot(time_axis, reconstructed[idx].flatten(), label='Clean EEG reconstruction', linestyle='--',linewidth=line_width)
        #plt.title(f'{title} Data at segment {idx+1}')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Normalized amplitude', fontsize=14)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
   

    plt.tight_layout(rect=[0, 0.1, 1, 1]) 
    
    plt.show()
    
# def show_unfiltering_images(original, reconstructed, idx, title):
#     fs = 200 
#     time_axis= np.linspace(0, 4, fs * 4, endpoint=False)  
#     plt.figure(figsize=(18,7))
#     # Plot original data
#     plt.subplot(1, 2 , 1)
#     plt.plot(time_axis, original[idx].flatten())
#     plt.title(f'Original {title} Data at segment {idx}')
#     plt.xlabel('Time(s)')
#     plt.ylabel('Normalized Amplitude')
#     # Plot reconstructed data
#     plt.subplot( 1, 2 ,2)
#     plt.plot(time_axis, reconstructed[idx].flatten())
#     plt.title(f'Reconstructed {title} Data at segment {idx} without filtering')
#     plt.xlabel('Time(s)')
#     plt.ylabel('Normalized Amplitude')
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
   
    
    # Load data
    x_test_clean = np.load("C:/Users/lijianing/Desktop/Autoencoder-ae_model/run_model/x_test_clean1.npy")
    x_test_noisy = np.load("C:/Users/lijianing/Desktop/Autoencoder-ae_model/run_model/x_test_noisy1.npy")
    
    aa=np.zeros(x_test_clean.shape)
    aa=x_test_clean
    
    bb=np.zeros(x_test_noisy.shape)
    bb=x_test_noisy
    #x_test_clean = np.expand_dims(x_test_clean, axis=-1)
    fs=200
    
    # for i in range(len(x_test_clean)):
    #      data= bandpass_filtering(x_test_clean[i], 1, 50, 200, 4)
    #      x_test_clean[i]=data
    
    # for i in range(967,1712):
    #      data= highpass_filtering(x_test_noisy[i], 1, 200, 4)
    #      x_test_noisy[i]=data
         


    
    
    # EEG_clean_EOG = np.load('EEG_clean_EOG_bp.npy', allow_pickle=True)    #filtering 
    # EEG_noisy_EOG = np.load('EEG_noisy_EOG_bp.npy', allow_pickle=True)
    
    # Check data shape
    print("Original clean dataset shape(with noisy):", x_test_noisy.shape)
    # full dataset
    dataset = EEGDataset(x_test_noisy,x_test_clean, segment_length)
    for i in range(len(dataset.clean_segments)):

        data1 = bandpass_filtering(dataset.clean_segments[i], 1, 50, 200, 8)
        dataset.clean_segments[i]=data1

        data2 = bandpass_filtering(dataset.noisy_segments[i], 1, 50, 200, 8)
        dataset.noisy_segments[i]=data2
        

    train_size=int(0.6*len(dataset.noisy_segments))
    val_size=int(0.8*len(dataset.noisy_segments))
    indices = list(range(len(dataset.noisy_segments)))
    
    # Define fixed train and Subsets
    beforenormalize_train_noisy_segment = bb[:train_size]
    beforenormalize_test_noisy_segment = bb[val_size:]
    beforenormalize_train_noisy_segment = torch.tensor(beforenormalize_train_noisy_segment, dtype=torch.float32)
    beforenormalize_test_noisy_segment = torch.tensor(beforenormalize_test_noisy_segment, dtype=torch.float32)
    train_noisy_segment = dataset.noisy_segments[:train_size]
    train_clean_segment = dataset.clean_segments[:train_size]
    val_noisy_segment = dataset.noisy_segments[train_size:val_size]
    val_clean_segment = dataset.clean_segments[train_size:val_size]
    test_noisy_segment = dataset.noisy_segments[val_size:]
    test_clean_segment = dataset.clean_segments[val_size:]
    
    train_dataset = EEGDataset(train_noisy_segment, train_clean_segment,segment_length)
    # for i in range(len(train_dataset.clean_segments)):
   
    #     data1 = normalization(train_dataset.clean_segments[i])
    #     train_dataset.clean_segments[i]=data1
       
    #     data2 = normalization(train_dataset.noisy_segments[i])
    #     train_dataset.noisy_segments[i]=data2
        
    val_dataset = EEGDataset(val_noisy_segment, val_clean_segment,segment_length)
    # for i in range(len(val_dataset.clean_segments)):
       
    #     data1 = normalization(data1)
    #     val_dataset.clean_segments[i]=data1
     
    #     data2 = normalization(data2)
    #     val_dataset.noisy_segments[i]=data2
    test_dataset = EEGDataset(test_noisy_segment, test_clean_segment,segment_length)
    # for i in range(len(test_dataset.clean_segments)):
      
    #     data1 = normalization(data1)
    #     test_dataset.clean_segments[i]=data1
    
    #     data2 = normalization(data2)
    #     test_dataset.noisy_segments[i]=data2
    
    # # # Store all segments
    # train_segments = [train_dataset[i] for i in range(len(train_dataset))]
    # test_segments = [test_dataset[i] for i in range(len(test_dataset))]
    # print("Train segments saved as 'train_segments' with shape ",len(train_dataset))
    # print("Test segments saved as 'test_segments' with shape ",len(test_dataset))
    
    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    test_dataset_size = len(test_dataset)

    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Testing data size: {len(test_dataset)}")
# metric
#%%
    encoder = Encoder(input_dim=x_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, dropout_rate=dropout_rate)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim2=hidden_dim2, hidden_dim1=hidden_dim1, output_dim=x_dim, dropout_rate=dropout_rate)
    model= Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    
    reconstructed_train_data_epoch, reconstructed_val_data_epoch,mse_train_list, RRMSEt_train_list, mse_val_list, RRMSEt_val_list, train_loss_list, val_loss_list = train_vae()
    all_test_reconstructed_segments, mse_list ,RRMSEt_list = test_vae()
    
    
#%% clean dataset metric---- only analyze test dataset
    
    #print("*MSE =", "%.4f" % np.mean(mse_list))

    clean_inputs_PSD_RRMSE=[]
    clean_inputs_PSD_RRMSEABS=[]
    nperseg = 200
    nfft=800
    PSD_len= nfft//2+1
    clean_inputs_PSD = np.zeros(shape=(len(test_clean_segment), PSD_len))
    clean_outputs_PSD = np.zeros(shape=(len(test_clean_segment), PSD_len))
    
    for i in range(len(test_clean_segment)):
        # 1. PSD input clean EEG
        f, pxx = signal.welch(test_clean_segment[i], fs=200, nperseg=nperseg, nfft=nfft) 
        clean_inputs_PSD[i] = pxx
        
        # 2. PSD denoised/reconstructed EEG
        f, pxx = signal.welch(all_test_reconstructed_segments[i], fs=200, nperseg=nperseg, nfft=nfft) 
        clean_outputs_PSD[i] = pxx  
    
    for i in range(len(test_clean_segment)):
        clean_inputs_PSD_RRMSE.append(RRMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))
        clean_inputs_PSD_RRMSEABS.append(RMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))

   ## 3. CC
    clean_inputs_CC = []
    for i in range(len(test_clean_segment)):
       result = scipy.stats.pearsonr(test_clean_segment[i], all_test_reconstructed_segments[i])    # Pearson's r
       clean_inputs_CC.append(result.statistic)


#%%  train dataset
##rrmsef

    train_inputs_PSD_RRMSE=[]
    train_inputs_PSD_RRMSEABS=[]
    nperseg = 200
    nfft=800
    PSD_len= nfft//2+1
    train_inputs_PSD = np.zeros(shape=(len(train_clean_segment), PSD_len))
    train_outputs_PSD = np.zeros(shape=(len(train_clean_segment), PSD_len))
    
    for i in range(len(train_clean_segment)):
        # 1. PSD input EEG
        f, pxx = signal.welch(train_clean_segment[i], fs=200, nperseg=nperseg, nfft=nfft) 
        train_inputs_PSD[i] = pxx
        
        # 2. PSD denoised/reconstructed EEG
        f, pxx = signal.welch(reconstructed_train_data_epoch[i], fs=200, nperseg=nperseg, nfft=nfft) 
        train_outputs_PSD[i] = pxx  
    
    for i in range(len(train_clean_segment)):
        train_inputs_PSD_RRMSE.append(RRMSE(train_inputs_PSD[i], train_outputs_PSD[i]))
        train_inputs_PSD_RRMSEABS.append(RMSE(train_inputs_PSD[i], train_outputs_PSD[i]))
        

    
   ## 3. CC
    train_inputs_CC = []
    for i in range(len(train_clean_segment)):
       result = scipy.stats.pearsonr(train_clean_segment[i], reconstructed_train_data_epoch[i])    # Pearson's r
       train_inputs_CC.append(result.statistic)
       
#%%  #%%  val dataset
##rrmsef

    val_inputs_PSD_RRMSE=[]
    val_inputs_PSD_RRMSEABS=[]
    nperseg = 200
    nfft=800
    PSD_len= nfft//2+1
    val_inputs_PSD = np.zeros(shape=(len(val_clean_segment), PSD_len))
    val_outputs_PSD = np.zeros(shape=(len(val_clean_segment), PSD_len))
    
    for i in range(len(val_clean_segment)):
        # 1. PSD input EEG
        f, pxx = signal.welch(val_clean_segment[i], fs=200, nperseg=nperseg, nfft=nfft) 
        val_inputs_PSD[i] = pxx
        
        # 2. PSD denoised/reconstructed EEG
        f, pxx = signal.welch(reconstructed_val_data_epoch[i], fs=200, nperseg=nperseg, nfft=nfft) 
        val_outputs_PSD[i] = pxx  
    
    for i in range(len(val_clean_segment)):
        val_inputs_PSD_RRMSE.append(RRMSE(val_inputs_PSD[i], val_outputs_PSD[i]))
        val_inputs_PSD_RRMSEABS.append(RMSE(val_inputs_PSD[i], val_outputs_PSD[i]))
        

    
   ## 3. CC
    val_inputs_CC = []
    for i in range(len(val_clean_segment)):
       result = scipy.stats.pearsonr(val_clean_segment[i], reconstructed_val_data_epoch[i])    # Pearson's r
       val_inputs_CC.append(result.statistic)
       






#%%
    whole_mse_list = np.concatenate([mse_train_list, mse_val_list, mse_list]) 

    
    whole_RRMSEt_list = np.concatenate([RRMSEt_train_list, RRMSEt_val_list, RRMSEt_list])

          
    whole_RRMSEf_list = train_inputs_PSD_RRMSE + val_inputs_PSD_RRMSE + clean_inputs_PSD_RRMSE
    

    whole_CC_list = train_inputs_CC + val_inputs_CC +  clean_inputs_CC

    

    reconstruted_data = torch.cat((reconstructed_train_data_epoch, reconstructed_val_data_epoch,all_test_reconstructed_segments), dim=0)
    #%%  artifact-free

    clean_detect = []
    noisy_detect = []
    clean_inputs = []
    clean_outputs = []
    
    CC_detectClean = np.zeros(shape=(len(dataset.clean_segments),1))
    for i in range(len(dataset.clean_segments)):
        # calculate cc between noisy and clean version, if they are similar, it means the noisy input is clean
        # if input test_clean and test_noisy is quite different, it means the signal is noisy
        CC_detectClean[i] = np.corrcoef(dataset.clean_segments[i], dataset.noisy_segments[i])[0,1]
        if CC_detectClean[i]>0.95:
            clean_detect.append(i)
        else:
            noisy_detect.append(i)
            
            
    MSE_clean = []
    RRMSEt_clean = []
    RRMSEf_clean = []
    CC_clean = []
      
    MSE_EOG = []
    RRMSEt_EOG = []
    RRMSEf_EOG = []
    CC_EOG = []
    
    MSE_motion = []
    RRMSEt_motion = []
    RRMSEf_motion = []
    CC_motion = []
    
    MSE_EMG = []
    RRMSEt_EMG = []
    RRMSEf_EMG = []
    CC_EMG = []

            
    for i in range(len(clean_detect)):
            MSE_clean.append(whole_mse_list[clean_detect[i]])
            RRMSEt_clean.append(whole_RRMSEt_list[clean_detect[i]])
            RRMSEf_clean.append(whole_RRMSEf_list[clean_detect[i]])
            CC_clean.append(whole_CC_list[clean_detect[i]])
        

    
    for i in range(len(noisy_detect)):
        
        if noisy_detect[i]<345:
            MSE_EOG.append(whole_mse_list[noisy_detect[i]])
            RRMSEt_EOG.append(whole_RRMSEt_list[noisy_detect[i]])
            RRMSEf_EOG.append(whole_RRMSEf_list[noisy_detect[i]])
            CC_EOG.append(whole_CC_list[noisy_detect[i]])
        elif noisy_detect[i]>=345 and noisy_detect[i]<967:
            MSE_motion.append(whole_mse_list[noisy_detect[i]])
            RRMSEt_motion.append(whole_RRMSEt_list[noisy_detect[i]])
            RRMSEf_motion.append(whole_RRMSEf_list[noisy_detect[i]])
            CC_motion.append(whole_CC_list[noisy_detect[i]])
        elif noisy_detect[i]>=967:
            MSE_EMG.append(whole_mse_list[noisy_detect[i]])
            RRMSEt_EMG.append(whole_RRMSEt_list[noisy_detect[i]])
            RRMSEf_EMG.append(whole_RRMSEf_list[noisy_detect[i]])
            CC_EMG.append(whole_CC_list[noisy_detect[i]])


 #%%   
    
    print("\n EEG clean input results: ")
    print("*RRMSE-Time: mean= ", "%.4f" % np.mean(RRMSEt_clean)," ,std= ", "%.4f" % np.std(RRMSEt_clean))
    print("*RRMSE-Freq: mean= ","%.4f" % np.mean(  RRMSEf_clean), " ,std= ", "%.4f" % np.std(  RRMSEf_clean))
    print("*CC:         mean= ", "%.4f" % np.mean( CC_clean), " ,std= ", "%.4f" % np.std(CC_clean))
 
 
    print("\n EEG EOG artifacts results: ")
    print("*RRMSE-Time: mean= ", "%.4f" % np.mean(RRMSEt_EOG)," ,std= ", "%.4f" % np.std(RRMSEt_EOG))
    print("*RRMSE-Freq: mean= ","%.4f" % np.mean(RRMSEf_EOG), " ,std= ", "%.4f" % np.std(RRMSEf_EOG))
    print("*CC:         mean= ", "%.4f" % np.mean(CC_EOG), " ,std= ", "%.4f" % np.std(CC_EOG))
    
    print("\n EEG motion artifacts results:")
    print("*RRMSE-Time: mean= ", "%.4f" % np.mean(RRMSEt_motion)," ,std= ", "%.4f" % np.std(RRMSEt_motion))
    print("*RRMSE-Freq: mean= ","%.4f" % np.mean(RRMSEf_motion), " ,std= ", "%.4f" % np.std(RRMSEf_motion))
    print("*CC:         mean= ", "%.4f" % np.mean(CC_motion), " ,std= ", "%.4f" % np.std(CC_motion))
    
    print("\n EEG EMG artifacts results:")
    print("*RRMSE-Time: mean= ", "%.4f" % np.mean(RRMSEt_EMG)," ,std= ", "%.4f" % np.std(RRMSEt_EMG))
    print("*RRMSE-Freq: mean= ","%.4f" % np.mean(RRMSEf_EMG), " ,std= ", "%.4f" % np.std(RRMSEf_EMG))
    print("*CC:         mean= ", "%.4f" % np.mean(CC_EMG), " ,std= ", "%.4f" % np.std(CC_EMG))
   
#%%  ##loss curve
    plt.figure(figsize=(15, 5))
    plt.plot(range(1, epochs + 1),train_loss_list, label='Training Loss',)
    plt.plot(range(1, epochs + 1),val_loss_list, label='Validation Loss')
    plt.title('Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.show()
    
    
#%% Show images for some indices in train and test data
    indices_to_show = range(2)
    for i in indices_to_show:
        show_images(train_dataset.clean_segments, reconstructed_train_data_epoch, beforenormalize_train_noisy_segment,i, 'Train')
        show_images(  test_dataset.clean_segments, all_test_reconstructed_segments, beforenormalize_test_noisy_segment,i, 'Test')
        
#%% Show images constrast  
    indices = random.sample(clean_detect, 8)
   
    show_image_constrast(dataset.clean_segments,  reconstruted_data, indices, title='Eight 4 s examples of originally clean EEG reconstruction examples')


#%% show images EOG
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.gridspec import GridSpec
    
    # def show_image_constrast_EOG(corrupted, original, reconstructed, indices, title, fs=200):
    #     time_interval = 4.0 
    #     time_axis = np.linspace(0, time_interval, int(fs * time_interval), endpoint=False)   
    #     plt.figure(figsize=(18, 9))

    
    #     for i, idx in enumerate(indices):
    #         plt.subplot(4, 4, i * 2 + 1)
    #         plt.plot(time_axis, corrupted[idx].flatten(),label='Corrupted EOG', color='blue')
    #         #ax1.set_title(f'{title} Original Data at segment {idx+1}')
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Amplitude (μV)')
        

    #         plt.subplot(4, 4, i * 2 + 2)
    #         plt.plot(time_axis, original[idx].flatten(), label='Clean EOG input', color='green')
    #         plt.plot(time_axis, reconstructed[idx].flatten(), label='Reconstructed EOG', color='orange', linestyle='--')
            
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Amplitude (Comparison)')
    #     plt.tight_layout()
    #     plt.show()
    

    # indices = random.sample(range(0, 345), 8) 
    # show_image_constrast_EOG(aa, dataset.clean_segments,  reconstruted_data, indices, title='EEG')




#%%
# #%%   frequency

def plot_single_comparison(original_signal, reconstructed_signal, fs):

    n_samples = len(original_signal)
    original_fft = np.fft.fft(original_signal)
    original_freqs = np.fft.fftfreq(n_samples, d=1/fs)
    positive_freqs = original_freqs[:n_samples // 2]
    positive_original_fft = np.abs(original_fft[:n_samples // 2])


    reconstructed_fft = np.fft.fft(reconstructed_signal)
    positive_reconstructed_fft = np.abs(reconstructed_fft[:n_samples // 2])


    plt.figure(figsize=(9, 6))
   
    plt.plot(positive_freqs, positive_original_fft, label='Original Signal')
    plt.plot(positive_freqs, positive_reconstructed_fft, label='Reconstructed Signal', linestyle='--')
    
    plt.title('Frequency Spectrum Comparison', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    #plt.grid(True)
    plt.legend()
    plt.show()

index = 3

original_signal = train_dataset.clean_segments[index]
reconstructed_signal_1 = reconstructed_train_data_epoch[index]

fs = 200  # 200 Hz

plot_single_comparison(original_signal, reconstructed_signal_1, fs)
