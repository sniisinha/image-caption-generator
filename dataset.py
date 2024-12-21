from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, caption_dict, features, tokenizer, max_length):
        self.caption_dict = caption_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keys = list(caption_dict.keys())
        self.features = features
        self.current_caption_index = 0

    def __len__(self):
        return len(self.caption_dict)

    def __getitem__(self, idx):
        image_key = self.keys[idx]
        captions = self.caption_dict[image_key]

        current_caption = captions[self.current_caption_index]

        seq = self.tokenizer.texts_to_sequences([current_caption])[0]

        image_tensor = self.features[image_key].squeeze().numpy()

        temp = torch.Tensor(self.max_length)
        input_sequence = torch.nn.utils.rnn.pad_sequence((torch.tensor(seq), temp), batch_first=True, padding_value=0)[0]

        self.current_caption_index = (self.current_caption_index + 1) % len(captions)

        return image_tensor, input_sequence