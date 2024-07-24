# I can listen but cannot read: An evaluation of two-tower multimodal systems for instrument recognition
This repository is the official implementation of 'I can listen but cannot read: An evaluation of two-tower multimodal systems for instrument recognition', accepted to ISMIR 2024.

In this paper we tried to pinpoint specific problems that arise in music-text multimodal systems, as well as try to estimate if the problems can be attributed mainly in the Audio or Text branch. 

## Introduction and general information
Two-tower systems use separate encoders for each modality in order to obtain vector representations (embeddings). In our case and for music-text multimodal systems, we have a music and a text encoder separately (e.g. HT-SAT and RoBERTa for LAION-CLAP). Fundamentally the embeddings from each respective modality (one for the chunk of audio and one for the sentence) are not directly comparable as they are of different dimensionality. In order to enable comparisons between them, we need to either:

1. Map the audio embedding to the text embedding space (change audio embedding dimensionality to text embedding dimensionality)
2. Map the text embedding to the audio embedding space (change text embedding dimensionality to audio embedding dimensionality)
3. Map both of them in a separate space (our case). These systems are usually referred to joint audio-text models.

In order to do this, we need to obtain pairs of sentences and audio and then force their embedding to be mapped close with each other. This is successfully done via Contrastive Learning, forcing these pairs to be close, while pushing away embeddings from any other combination of audio and caption in the batch. The latter, are not a part of the dataset and are referred to as negative pairs.

We perform several tests for both the embeddings either obtained straight from the encoder or after being mapped to the joint space.

We trained MusCALL using repository (ADD ILLARIAS REPOSITORY) and the default hyper-parameters. As a dataset, we used LPMusicCaps-MTT (ADD CITATION AND LINK). If you need the model please contact the author (see email below).

Apart from that, we downloaded and used the LAION-CLAP models provided in (ADD LINK TO LAION CLAP).

We used Doktorski similarity to obtain triplets (anchor, positive, negative) of musical instrument terminology and checked if our models could successfuly evaluate that anchor is closer to positive rather than negative.

## Setup - Initial steps

### Create a new conda environment with a name \<env\> of your choice
```
conda create --name <env> --file requirements.txt
```

### Download the models. We tested:
1. [music_audioset_epoch_15_esc_90.14.pt](https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt?download=true)
2. [music_speech_audioset_epoch_15_esc_89.98.pt](https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt?download=true)
3. MusCALL trained on LPMusicCaps-MTT

and move them in the checkpoints folder.

### Generate the Tinysol Doktorski based similarity and triplets for evaluation

```
python evaluation_set_generation.py --evaluation_type <type>
```

where choose *all* if you want to use the full Doktorski ontology or *tinysol* for just the tinysol terms inside the doktorski similariy.

### Generate the TinySol embeddings
Download the [TinySOL dataset](https://forum.ircam.fr/projects/detail/tinysol/) and move every song (not the folders) in TinySol folder. Then, generate all the embeddings and save them as .npy files using:

```
python embeddings_generations.py
```

### Evaluation
Evaluate the embeddings obtained from the models with a similarity function of your choice. We used cosine similarity.

## Citation

```

```

## Questions
For any questions or to have access to a trained MusCall model, send an email to **i.vasilakis@qmul.ac.uk**