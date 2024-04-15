#!/usr/bin/env python
# coding: utf-8

# ## New approach - We'll only consider only those movies which are in the user's history and train the Transformer using those movies instead of considering all the movies in the genre

# In[205]:


import pandas as pd
import numpy as np
import os
import re
import glob


# In[206]:


from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


# In[207]:


def smart_search(folder_path, movie_names):
    # Get a list of all files in the specified folder
    all_files = os.listdir(folder_path)

    # Initialize an empty list to store the matching file names
    matching_files = []

    # Loop through each movie name entered by the user
    for movie_name in movie_names:
        
        # Convert movie name to lowercase for case-insensitive comparison
        movie_name_lower = movie_name.lower()

        # Create a pattern to match the file name
        pattern = f"{movie_name_lower} "

        # Use list comprehension to find matching files
        matching_files.extend([file for file in all_files if pattern in file.lower()])
        


        

    return matching_files[:len(movie_names)]


# In[208]:


# Example usage:
folder_path = "/Users/nirupam/Library/Mobile Documents/com~apple~CloudDocs/3-2_Project/3-2_Project_Dataset/2_reviews_per_movie_raw"
# user_input_movies = [
#     "3 IDIOTS",
#     "8 Mile",
#     "42",
#     "All the President's Men", 
#     "Iron Man"
# ]

user_input_movies = ['The Hurricane', "Rosemary's Baby", 'The patriot', 'Duck You Sucker', 'Heat', "Mr. Holland's Opus", 'The Untouchables', 'Forrest Gump', 'Rocky II', 'Casablanca']


# In[209]:


user_found_movies = smart_search(folder_path, user_input_movies)
if user_found_movies:
    print("Matching files found:")
    for file_name in user_found_movies:
        print(file_name)
else:
    print("No matching files found.")


# In[210]:


user_found_movies


# In[211]:


# combined_reviews_user = pd.DataFrame(columns=['Movie', 'Reviews', 'Helpful_Reviews'])

# for movie_name in user_found_movies:
#     csv_file_path = f'/Users/nirupam/Library/Mobile Documents/com~apple~CloudDocs/3-2_Project/3-2_Project_Dataset/2_reviews_per_movie_raw/{movie_name}'
#     if os.path.isfile(csv_file_path):
#         reviews_df = pd.read_csv(csv_file_path)
#         reviews_df.drop(['username', 'rating', 'total', 'date', 'title'], axis=1, inplace=True)
#         all_reviews = ' '.join(reviews_df['review'])
#         sorted_reviews = reviews_df.sort_values('helpful', ascending=False)

#         # Take top 3 helpful reviews
#         top_3_reviews = sorted_reviews.head(3)

#         # Create a new DataFrame with the current movie's data for each of the top 3 reviews
#         for i in range(3):
#             new_df = pd.DataFrame({'Movie': [movie_name], 'Reviews': [all_reviews], 'Helpful_Reviews': [top_3_reviews.iloc[i]['review']]})
#             combined_reviews_user = pd.concat([combined_reviews_user, new_df], ignore_index=True)

# # Drop the `Movie` column from `combined_reviews_user`
# combined_reviews_user = combined_reviews_user.drop(['Movie'], axis=1)

# combined_reviews_user

import os
import pandas as pd
import numpy as np

combined_reviews_user = pd.DataFrame(columns=['Movie', 'Reviews', 'Helpful_Reviews'])

for movie_name in user_found_movies:
    csv_file_path = f'/Users/nirupam/Library/Mobile Documents/com~apple~CloudDocs/3-2_Project/3-2_Project_Dataset/2_reviews_per_movie_raw/{movie_name}'
    if os.path.isfile(csv_file_path):
        reviews_df = pd.read_csv(csv_file_path)
        reviews_df.drop(['username', 'rating', 'total', 'date', 'title'], axis=1, inplace=True)

        all_reviews = ' '.join(reviews_df['review'])
        sorted_reviews = reviews_df.sort_values('helpful', ascending=False)

        # Take top 3 helpful reviews
        top_3_reviews = sorted_reviews.head(3)

        # Exclude top 3 helpful reviews from the "Reviews" column
        remaining_reviews = reviews_df[~reviews_df['review'].isin(top_3_reviews['review'])]['review']

        # Divide the remaining reviews into 3 parts
        num_reviews = len(remaining_reviews)
        chunk_size = num_reviews // 3

        # Create a new DataFrame with the current movie's data for each chunk of reviews
        for i in range(3):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 2 else num_reviews
            chunk_reviews = ' '.join(remaining_reviews.iloc[start_idx:end_idx])

            new_df = pd.DataFrame({'Movie': [movie_name], 'Reviews': [chunk_reviews], 'Helpful_Reviews': [top_3_reviews.iloc[i]['review']]})
            combined_reviews_user = pd.concat([combined_reviews_user, new_df], ignore_index=True)

# Drop the `Movie` column from `combined_reviews_user`
combined_reviews_user = combined_reviews_user.drop(['Movie'], axis=1)



# ### Combining all reviews of all movies given by the user

# In[212]:


# combined_reviews_user = pd.DataFrame(columns=['Movie', 'Reviews', 'Helpful_Reviews'])

# for movie_name in user_found_movies:
#     csv_file_path = f'/Users/nirupam/Library/Mobile Documents/com~apple~CloudDocs/3-2_Project/3-2_Project_Dataset/2_reviews_per_movie_raw/{movie_name}'
#     if os.path.isfile(csv_file_path):
#         reviews_df = pd.read_csv(csv_file_path)
#         reviews_df.drop(['username', 'rating', 'total', 'date', 'title'], axis=1, inplace=True)
#         all_reviews = ' '.join(reviews_df['review'])
#         sorted_reviews = reviews_df.sort_values('helpful', ascending=False)
#         top_5_reviews = '. '.join(sorted_reviews.head(5)['review'])
#         remaining_reviews = sorted_reviews.iloc[5:]['review']
#         remaining_reviews_concatenated = ' '.join(remaining_reviews)

#         # Create a new DataFrame with the current movie's data
#         new_df = pd.DataFrame({'Movie': [movie_name], 'Reviews': [remaining_reviews_concatenated], 'Helpful_Reviews': [top_5_reviews]})

#         # Append the new DataFrame to `combined_reviews_user`
#         combined_reviews_user = pd.concat([combined_reviews_user, new_df], ignore_index=True)

# # Drop the `Movie` column from `combined_reviews_user`
# combined_reviews_user = combined_reviews_user.drop(['Movie'], axis=1)




# In[213]:


combined_reviews_user


# In[214]:


re_pattern = re.compile('[<br/>$@&?]')

combined_reviews_user['Reviews'] = combined_reviews_user['Reviews'].str.replace(re_pattern, '')
combined_reviews_user['Helpful_Reviews'] = combined_reviews_user['Helpful_Reviews'].str.replace(re_pattern, '')


# In[215]:


class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, decodings):
        self.encodings = encodings
        self.decodings = decodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.decodings['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts=None, val_labels=None, 
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = PegasusTokenizer.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    decodings = tokenizer(labels, truncation=True, padding=True, return_tensors="pt")
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer
    
    
def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=25, # 100          # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=1,    # batch size for evaluation, can increase if memory allows
      save_steps= 500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1, # 100          # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      save_steps=1000,                  # number of updates steps before checkpoint saves
      save_total_limit=2,  # 5            # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

    


# In[216]:


# Extract the input and output columns from the dataframe
train_texts = combined_reviews_user["Reviews"]
train_labels = combined_reviews_user["Helpful_Reviews"]


# In[217]:


# Specify the model name and prepare the data
model_name = 'google/pegasus-xsum'
train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts.tolist(), train_labels.tolist())


# In[218]:


# Prepare the fine-tuning trainer
# uncomment this to start training/fine-tuning
trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
trainer.train()


# In[219]:


#output_dir = "/home/u20010/Sem_Project/fine_tune_1"
output_dir = "/Users/nirupam/Desktop/Sem_Project/fine_tuned/"
trainer.save_model(output_dir)


# In[220]:


model_1 = PegasusForConditionalGeneration.from_pretrained(output_dir)


# In[229]:


# input_text = '''In college, Farhan and Raju form a great bond with Rancho due to his refreshing outlook. Years later, a bet gives them a chance to look for their long-lost friend whose existence seems rather elusive.'''
# input_text = '''Rancho is an engineering student. His two friends. Farhan and Raju, Rancho sees the world in a different way. Rancho goes somewhere one day. And his friends find him. When Rancho is found, he has become one of a great scientist in the world'''

## input_text = '''During the British era, Malli, a small tribal girl, is taken away by British governor Scott Buxton and his wife Catherine against the wishes of her mother. Rama Raju is an Indian cop who works for the British army; for him duty comes first, and he is very ruthless to revolutionary Indians but is never given his due by British government. The British government find that a tribal Komaram Bheem, who considers Malli his sister, has started his search for her and could be an obstacle for the British army. The governor and his wife announce a special post for any officer who can bring Bheem to them. Rama Raju decides to take the matters into his own hands and promises the government to bring him in dead or alive. Bheem by now has reached he city in search of Malli and pretends to be a mechanic, Akhtar. During a train accident on a lake he and Rama Raju risk their lives and save a kid and become best of friends. But each man will clash with the other and will thirst for each other's blood in order to complete their missions.'''

## input_text = '''It is 1942, America has entered World War II, and sickly but determined Steve Rogers is frustrated at being rejected yet again for military service. Everything changes when Dr. Erskine recruits him for the secret Project Rebirth. Proving his extraordinary courage, wits and conscience, Rogers undergoes the experiment and his weak body is suddenly enhanced into the maximum human potential. When Dr. Erskine is then immediately assassinated by an agent of Nazi Germany's secret HYDRA research department (headed by Johann Schmidt, a.k.a. the Red Skull), Rogers is left as a unique man who is initially misused as a propaganda mascot; however, when his comrades need him, Rogers goes on a successful adventure that truly makes him Captain America, and his war against Schmidt begins.'''

## input_text = '''After the events of Captain America: Civil War, Prince T'Challa returns home to the reclusive, technologically advanced African nation of Wakanda to serve as his country's new king. However, T'Challa soon finds that he is challenged for the throne from factions within his own country. When two foes conspire to destroy Wakanda, the hero known as Black Panther must team up with C.I.A. agent Everett K. Ross and members of the Dora Milaje, Wakandan special forces, to prevent Wakanda from being dragged into a world war.'''

## input_text = '''84 years later, a 100 year-old woman named Rose DeWitt Bukater tells the story to her granddaughter Lizzy Calvert, Brock Lovett, Lewis Bodine, Bobby Buell and Anatoly Mikailavich on the Keldysh about her life set in April 10th 1912, on a ship called Titanic when young Rose boards the departing ship with the upper-class passengers and her mother, Ruth DeWitt Bukater, and her fiancé, Caledon Hockley. Meanwhile, a drifter and artist named Jack Dawson and his best friend Fabrizio De Rossi win third-class tickets to the ship in a game. And she explains the whole story from departure until the death of Titanic on its first and last voyage April 15th, 1912 at 2:20 in the morning'''

input_text = '''Paleontologists Alan Grant and Ellie Sattler and mathematician Ian Malcolm are among a select group chosen to tour an island theme park populated by dinosaurs created from prehistoric DNA. While the park's mastermind, billionaire John Hammond, assures everyone that the facility is safe, they find out otherwise when various ferocious predators break free and go on the hunt.'''


input_text = input_text.replace("<br/><br/>", " ")
encoded_input = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
input_token_length = len(tokenizer.encode(input_text, max_length=None, return_tensors='pt')[0])
print(input_token_length)
print()

if input_token_length < 100:
    max_length_factor = 0.6  # You can adjust this factor based on your preference
    min_length_factor = 0.4 # You can adjust this factor based on your preference

else:
    max_length_factor = 0.5 #0.6 # You can adjust this factor based on your preference
    min_length_factor = 0.3  # You can adjust this factor based on your preference

max_length = int(input_token_length * max_length_factor)
min_length = int(input_token_length * min_length_factor)
summary_ids = model_1.generate(encoded_input['input_ids'], max_length=max_length, min_length=min_length)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
summary


# In[230]:


tokenizer_origi = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# create a model
model_origi = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

encoded_input_origi = tokenizer_origi(input_text, truncation=True, padding=True, return_tensors="pt")

input_token_length_origi = len(tokenizer_origi.encode(input_text, max_length=None, return_tensors='pt')[0])
print(input_token_length_origi)
print()
summary_vector = model_origi.generate(encoded_input_origi['input_ids'], max_length=max_length, min_length=min_length)

summary_origi = tokenizer_origi.decode(summary_vector[0], skip_special_tokens=True)

summary_origi


# In[ ]:




