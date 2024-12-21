import string

def preprocess_captions(data_path):
    caption_dict = {}
    word_to_index = {"<pad>": 0, "<start>": 1, "<end>": 2}
    index = 3 

    with open(data_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue
        image_name, caption = tokens[0], ' '.join(tokens[1:])

        caption = ' '.join([word.lower() for word in caption.split() if len(word) > 1])

        caption = clean_caption(caption)

        for word in caption.split():
            if word not in word_to_index:
                word_to_index[word] = index
                index += 1

        caption = '<start> ' + caption + ' <end>'

        if image_name in caption_dict.keys():
            caption_dict[image_name].append(caption)
        else:
            caption_dict[image_name] = [caption]

    index_to_word = {index: word for word, index in word_to_index.items()}

    return caption_dict, word_to_index, index_to_word

def clean_caption(caption):
    cleaned_caption = ""
    caption = ''.join([char for char in caption if char not in string.punctuation])
    for word in caption.split():
        if len(word) > 1:
            if word.isalpha():
                cleaned_caption += " " + word
                
    return cleaned_caption

def get_captions(caption_dict):
    captions = [caption for captions_list in caption_dict.values() for caption in captions_list]
    return captions

def get_max_length(caption_dict):
    max_length = 0
    for captions in caption_dict.values():
        for caption in captions:
            caption_length = len(caption.split())
            if caption_length > max_length:
                max_length = caption_length
    return max_length
