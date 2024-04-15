# Abstractive-Text-Summarization

## Introduction

This project aims to enhance the movie-watching experience by providing personalized movie summaries based on the user's viewing history. It utilizes abstractive text summarization, employing the fine-tuned PEGASUS model developed by Google with the IMDB dataset. By considering the user's ten previous movies, the summary for their next movie choice is customized to their preferences, incorporating elements from their past viewing experiences. This personalized approach improves decision-making and ensures the user receives summaries aligned with their interests.

## Dataset
To implement personalized movie summarization based on user viewing history, we utilized the IMDb Movie Reviews dataset, which can be accessed from the provided link (https://ieee-dataport.org/open-access/imdb-movie-reviews-dataset). This dataset consists of nearly 1 million unique movie reviews from 1150 different IMDb movies spread across 17 IMDb genres, including Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Fantasy, History, Horror, Music, Mystery, Romance, Sci-Fi, Sport, Thriller, and War. Each movie entry in the dataset also contains additional metadata such as the date of release, run length, IMDb rating, movie rating (PG-13, R, etc.), number of IMDb raters, and the number of reviews per movie.

Download the dataset from the link specified above to run the codeon your personal computer.

## User Interface using a Third party website .

Designed a User Interface using a third party website where we input both the history of user as well as the current movie storyline to get a neat and concise summary.

![image](https://github.com/RudrarajuAsrithVarma/Abstractive-Text-Summarization/assets/98108770/42d194a6-6205-421c-bb04-f3e2440b6568)

## Results and Validation

To know if the model generated summary is relevant or not, we have conducted an online survey using Microsoft forms where we asked the people to read both the surveys the input one and the model generated one, then we have asked them two important questions of Yes/No answer type. These are the statistics of the answers,

![image](https://github.com/RudrarajuAsrithVarma/Abstractive-Text-Summarization/assets/98108770/918f2448-eb3b-45c2-8088-a524dabcdf18)

![image](https://github.com/RudrarajuAsrithVarma/Abstractive-Text-Summarization/assets/98108770/024e614a-a4ad-4f1b-a444-3144d7024aad)

## We Used a python code to check for similarity between the generated summaries and the results are as follows:-

![image](https://github.com/RudrarajuAsrithVarma/Abstractive-Text-Summarization/assets/98108770/6ee70679-56bd-4efd-ae21-795501281664)

By this method we’ll get similar words percentage in both the summaries i.e., the input and the generated summaries. The less the similarity percentage the better the model in terms of abstractive accuracy as we’re not using the same words again from the input summary in the model generated summary i.e., contrary to extractive text summarization where the model has to generate a summary with most of the words taken from the input and making it shorter and preserving the overall meaning and context.

## Conclusion
In conclusion, the abstractive summarization project aimed to generate concise and meaningful summaries using advanced natural language processing techniques. Through the utilization of models like Pegasus-XSum and leveraging the power of NVIDIA DGX-1 supercomputer, we were able to fine-tune large-scale models that are computationally intensive to run on personal computers. This enabled us to generate abstractive summaries that captured the essence of the original text in a more human-like manner.




