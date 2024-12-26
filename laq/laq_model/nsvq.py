## https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization/blob/main/NSVQ.py
## NSVQ: Noise Substitution in Vector Quantization for Machine Learning in IEEE Access journal, January 2022

import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist

## add project_in, project_out layer 
## FYI vector_quantize_pytorch
class NSVQ(torch.nn.Module):
    def __init__(self, dim, num_embeddings, embedding_dim, device=torch.device('cpu'), discarding_threshold=0.1, initialization='normal', code_seq_len=1, patch_size=32, image_size = 256):
        super(NSVQ, self).__init__()

        """
        Inputs:
        
        1. num_embeddings = Number of codebook entries
        
        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)
        
        3. device = The device which executes the code (CPU or GPU)
        
        ########## change the following inputs based on your application ##########
        
        4. discarding_threshold = Percentage threshold for discarding unused codebooks
        
        5. initialization = Initial distribution for codebooks

        """
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size

        if initialization == 'normal':
            codebooks = torch.randn(self.num_embeddings, self.embedding_dim, device=device)
        elif initialization == 'uniform':
            codebooks = uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings).sample([self.num_embeddings, self.embedding_dim])
        else:
            raise ValueError("initialization should be one of the 'normal' and 'uniform' strings")

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains the number of times each codebook is used
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=device)
        
        self.project_in = torch.nn.Linear(dim, embedding_dim)
        self.project_out = torch.nn.Linear(embedding_dim, dim) 
        
        # 8 * 8  => 4 * 4 => 2 * 2
        #assert patch_size == 32
        if code_seq_len == 1:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=4, stride=1, padding=0),
            )
        elif code_seq_len == 2:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=(3, 4), stride=1, padding=0),
            )
            
        elif code_seq_len == 4:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=1, padding=0),
            )
        elif code_seq_len == 16:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            )
        elif code_seq_len == 64:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            )
        elif code_seq_len == 256:
            self.cnn_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            )
        else:
            raise ValueError("Not Implement: code_seq_len should be one of the 1 and 4 integers")

    def encode(self, input_data, batch_size):
        # compute the distances between input and codebooks vectors
        input_data = self.project_in(input_data) # b * 64 * 32
        # change the order of the input_data to b * 32 * 64
        input_data = input_data.permute(0, 2, 1).contiguous()
        # reshape input_data to 4D b*h*w*d
        input_data = input_data.reshape(batch_size, self.embedding_dim, int(self.image_size/self.patch_size), int(self.image_size/self.patch_size))
        input_data = self.cnn_encoder(input_data) # 1*1 tensor
        input_data = input_data.reshape(batch_size, self.embedding_dim, -1) # b * 32 * d^2
        input_data = input_data.permute(0, 2, 1).contiguous() # b * 1 * 32
        input_data = input_data.reshape(-1, self.embedding_dim)
        return input_data
    
    def decode(self, quantized_input, batch_size):
        quantized_input = quantized_input.reshape(batch_size, self.embedding_dim, -1) # b * 32 * d^2
        quantized_input = quantized_input.permute(0, 2, 1).contiguous() # b * 64 * 32
        
        quantized_input = self.project_out(quantized_input)
        return quantized_input
    
    def forward(self, input_data_first, input_data_last, codebook_training_only=False):

        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """

        batch_size = input_data_first.shape[0]
        
        input_data_first = input_data_first.contiguous()
        
        input_data_first = self.encode(input_data_first, batch_size) # b * 1 * 32
        input_data_last = self.encode(input_data_last, batch_size) # b * 1 * 32
        
        input_data = input_data_last - input_data_first
           
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, self.codebooks.t()))
                     + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)
                
        hard_quantized_input = self.codebooks[min_indices]
        
        random_vector = normal_dist.Normal(0, 1).sample(input_data.shape).to(self.device)

        norm_quantization_residual = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_quantization_residual / norm_random_vector + self.eps) * random_vector

        if codebook_training_only:
            print(f"codebook error: {norm_quantization_residual.norm()}")
            quantized_input = hard_quantized_input
        else:
            quantized_input = input_data + vq_error

        # claculating the perplexity (average usage of codebook entries)
        encodings = torch.zeros(input_data.shape[0], self.num_embeddings, device=input_data.device)
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            min_indices_cpu = min_indices.cpu()
            self.codebooks_used[min_indices_cpu] += 1

        # use the first returned tensor "quantized_input" for training phase (Notice that you do not have to use the
        # tensor "quantized_input" for inference (evaluation) phase)
        # Also notice you do not need to add a new loss term (for VQ) to your global loss function to optimize codebooks.
        # Just return the tensor of "quantized_input" as vector quantized version of the input data.
        
        quantized_input = self.decode(quantized_input, batch_size)
        return quantized_input, perplexity, self.codebooks_used.cpu().numpy(), min_indices.reshape(batch_size, -1)

    def replace_unused_codebooks(self, num_batches):

        """
        This function is used to replace the inactive codebook entries with the active ones, to make all codebooks
        entries to be used for training. The function has to be called periodically with the periods of "num_batches".
        For more details, the function waits for "num_batches" training batches and then discards the codebook entries
        which are used less than a specified percentage (self.discard_threshold) during this period, and replace them
        with the codebook entries which were used (active).

        Recommendation: Call this function after a specific number of training batches. In the beginning the number of
         replaced codebooks might increase. However, the main trend must be decreasing after some training time.
         If it is not the case for you, increase the "num_batches" or decrease the "discarding_threshold" to make
         the trend for number of replacements decreasing. Stop calling the function at the latest stages of training
         in order not to introduce new codebook entries which would not have the right time to be tuned and optimized
         until the end of training.

        Play with "self.discard_threshold" value and the period ("num_batches") you call the function. A common trend
        could be to select the self.discard_threshold from the range [0.01-0.1] and the num_batches from the set
        {100,500,1000,...}. For instance, as a commonly used case, if we set the self.discard_threshold=0.01 and
        num_batches=100, it means that you want to discard the codebook entries which are used less than 1 percent
        during 100 training batches. Remember you have to set the values for "self.discard_threshold" and "num_batches"
        in a logical way, such that the number of discarded codebook entries have to be in a decreasing trend during
        the training phase.

        :param num_batches: period of training batches that you want to replace inactive codebooks with the active ones

        """

        with torch.no_grad():

            unused_indices = torch.where((self.codebooks_used.cpu() / num_batches) < self.discarding_threshold)[0]
            used_indices = torch.where((self.codebooks_used.cpu() / num_batches) >= self.discarding_threshold)[0]

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            if used_count == 0:
                print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                self.codebooks += self.eps * torch.randn(self.codebooks.size(), device=self.device).clone()
            else:
                used = self.codebooks[used_indices].clone()
                if used_count < unused_count:
                    used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                    used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                else:
                    used_codebooks = used

                self.codebooks[unused_indices] *= 0
                self.codebooks[unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                    (unused_count, self.embedding_dim), device=self.device).clone()

            print(f'************* Replaced ' + str(unused_count) + f' codebooks *************')
            self.codebooks_used[:] = 0.0    

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):

        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """

        #input_data = self.project_in(input_data)
        input_data_first = input_data_first.detach().clone()
        input_data_last = input_data_last.detach().clone()
        codebooks = self.codebooks.detach().clone()
        ###########################################
        
        batch_size = input_data_first.shape[0]
        # compute the distances between input and codebooks vectors
        
        input_data_first = self.encode(input_data_first, batch_size) # b * n * dim
        input_data_last = self.encode(input_data_last, batch_size) # b * n * dim
        
        input_data = input_data_last - input_data_first
                
        
        input_data = input_data.reshape(-1, self.embedding_dim)     

        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))

        min_indices = torch.argmin(distances, dim=1)
        
        if user_action_token_num is not None:
            if type(user_action_token_num) == list:
                min_indices = torch.tensor(user_action_token_num, device=self.device)
            else:                
                min_indices = torch.tensor([[user_action_token_num]], device=self.device).repeat(input_data.shape[0], 1)
        quantized_input = codebooks[min_indices]
        
        quantized_input = self.decode(quantized_input, batch_size)

        #use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input, min_indices.reshape(batch_size, -1)
    
    def codebook_reinit(self):
        self.codebooks = torch.nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim, device=self.device), requires_grad=True)
        self.codebooks_used = torch.zeros(self.num_embeddings, dtype=torch.int32, device=self.device)
        
        