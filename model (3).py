import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.hidden_size=hidden_size 
        self.embed_size=embed_size
        self.embeddings=nn.Embedding(vocab_size,embed_size)
        self.LSTM=nn.LSTM(embed_size,self.hidden_size,dropout=0.1,num_layers=num_layers)
        self.linear=nn.Linear(self.hidden_size,vocab_size)
        self.num_layers=num_layers
        
    
    def forward(self, features, captions):
        captions= self.embeddings(captions[:,:-1])
        batch_size=features.size(0)
        

        #self.current_state = torch.zeros((self.embed_size, batch_size, self.hidden_size))
        
        
        x=torch.cat((features.unsqueeze(1),captions),dim=1)
        output,self.hidden_state=self.LSTM(x)
        output=self.linear(output)
        return output 
                                     
        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output=[]
        hidden_state = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
            pred,hidden_state=self.LSTM(inputs,hidden_state)
            pred=self.linear(pred)
            pred=torch.argmax(pred,dim=2)
            predicted_index=pred.item()
            output.append(predicted_index)
            inputs = self.embeddings(pred)
        return output    
               
            
            