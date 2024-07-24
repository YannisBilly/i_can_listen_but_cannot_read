import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(prog = 'I can listen but cannot read',
                                 description = 'Oficial implementatino of paper accepted at ISMIR 2024')

parser.add_argument('-e', '--experiment', help = 'The experiment to be run.',
                    default='all')

args = parser.parse_args()

if args.experiment == 'zeroshot_baseline':
    # Zero-shot baseline with multiple prompts
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    DEFINITION_TYPES = ['muscall_prompt', 'definition', 'definition_no_label_word',
                        'label_random_context', 'negative_muscall_prompts',
                        'solo_musical_instrument_sound_of', 'this_is_a_recording_of']


    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')
    gt_labels = groundtruth.iloc[:, 1:].to_numpy()

    for model in MODELS:
        with open(f'evaluation_output/zeroshot_baseline/{model}.csv', 'w') as file:
            writer = csv.writer(file)

            writer.writerow(['model', 'prompt', 'top-1', 'top-2', 'top-3', 'top-4', 'top-5', 'F1', 'ROC-AUC', 'PR-AUC'])
            for def_type in DEFINITION_TYPES:
                row_to_write = []

                audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
                prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/{def_type}.npy')

                similarity = cosine_similarity(audio_embeddings, prompt_embeddings)

                row_to_write.append(model)
                row_to_write.append(def_type)

                for i in range(1,6):
                    row_to_write.append(top_k_accuracy_score(np.argmax(gt_labels, axis = -1), similarity, k = i))

                row_to_write.append(f1_score(np.argmax(gt_labels, axis = -1), np.argmax(similarity, axis = -1), average = 'weighted'))

                row_to_write.append(roc_auc_score(gt_labels, similarity))
                row_to_write.append(average_precision_score(gt_labels, similarity))

                writer.writerow(row_to_write)

elif args.experiment == 'audio_only':
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')

    for model in MODELS:
        with open(f'evaluation_output/audio_only/{model}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['model', 'top-1', 'top-2', 'top-3', 'top-4', 'top-5', 'F1', 'ROC-AUC', 'PR-AUC'])

            for space_type in ['joint_space', 'pre_joint']:
                audio_embeddings = np.load(f'model_evaluation/{model}/audio/{space_type}/tinysol.npy')
                gt_labels = groundtruth.iloc[:, 1:].columns.to_list()

                label_embeddings = []

                for label in gt_labels:
                    indexes = groundtruth[groundtruth[label] == 1].index.to_list()

                    mean_embedding = np.mean(audio_embeddings[indexes, :], axis = 0)
                    mean_embedding /= np.linalg.norm(mean_embedding)

                    label_embeddings.append(mean_embedding)

                label_embeddings = np.stack(label_embeddings, axis = 0)
                import pdb; pdb.set_trace()
                similarity = cosine_similarity(audio_embeddings, label_embeddings)

                row_to_write = []

                row_to_write.append(model)
                row_to_write.append(space_type)

                gt_labels = groundtruth.iloc[:, 1:].to_numpy()

                for i in range(1,6):
                    row_to_write.append(top_k_accuracy_score(np.argmax(gt_labels, axis = -1), similarity, k = i))

                row_to_write.append(f1_score(np.argmax(gt_labels, axis = -1), np.argmax(similarity, axis = -1), average = 'weighted'))

                row_to_write.append(roc_auc_score(gt_labels, similarity))
                row_to_write.append(average_precision_score(gt_labels, similarity))

                writer.writerow(row_to_write)

elif args.experiment == 'positive_negative_prompts':
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    DEFINITION_TYPES = ['muscall_prompt', 'definition', 'definition_no_label_word',
                        'label_random_context', 'negative_muscall_prompts',
                        'solo_musical_instrument_sound_of', 'this_is_a_recording_of']


    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')
    gt_labels = groundtruth.iloc[:, 1:].to_numpy()

    for model in MODELS:
        with open(f'evaluation_output/positive_negative/{model}.csv', 'w') as file:
            writer = csv.writer(file)

            writer.writerow(['model', 'prompt', 'top-1', 'top-2', 'top-3', 'top-4', 'top-5', 'F1', 'ROC-AUC', 'PR-AUC'])

            row_to_write = []

            audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
            positive_prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/muscall_prompt.npy')
            negative_prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/negative_muscall_prompts.npy')

            prompt_embeddings = np.concatenate((positive_prompt_embeddings, negative_prompt_embeddings), axis = 0)

            similarity = cosine_similarity(audio_embeddings, prompt_embeddings)

            row_to_write.append(model)
            row_to_write.append('positive_negative')

            for i in range(1,6):
                row_to_write.append(top_k_accuracy_score(np.argmax(gt_labels, axis = -1), similarity, k = i))

            row_to_write.append(f1_score(np.argmax(gt_labels, axis = -1), np.argmax(similarity, axis = -1), average = 'weighted'))

            row_to_write.append(roc_auc_score(gt_labels, similarity))
            row_to_write.append(average_precision_score(gt_labels, similarity))

            writer.writerow(row_to_write)

elif args.experiment == 'joint_doktorski_similarity':
    # Doktorski similarity in joint space
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    tinysol_instruments = ['bass tuba', 'french horn', 'trombone', 'trumpet', 'accordion', 'contrabass', 'violin', 'viola', 'cello', 'bassoon', 'clarinet', 'flute', 'oboe', 'alto saxophone']
    doktorski_sim = np.load('tinysol_instrument_similarity.npy')

    triplets = []
    triplets_indexes = []

    for anchor_index in range(len(tinysol_instruments)):
        for positive_index in range(anchor_index + 1, len(tinysol_instruments)):
            for negative_index in range(anchor_index + 1, len(tinysol_instruments)):
                if negative_index != positive_index:
                    if doktorski_sim[anchor_index,positive_index] > -1:
                        if doktorski_sim[anchor_index, positive_index] > doktorski_sim[anchor_index, negative_index]:
                            triplets.append([tinysol_instruments[anchor_index], tinysol_instruments[positive_index], tinysol_instruments[negative_index]])
                            triplets_indexes.append([anchor_index, positive_index, negative_index])


    for model in MODELS:
        text_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/muscall_prompt.npy')
        similarity = cosine_similarity(text_embeddings, text_embeddings)

        accuracy = 0
        for triplet in triplets_indexes:
            if similarity[triplet[0], triplet[1]] > similarity[triplet[0], triplet[2]]:
                accuracy += 1

        accuracy /= len(triplets_indexes)
        print(f'{model}: {accuracy}')

elif args.experiment == 'pre_joint_doktorski_similarity':
    # Doktorski similarity in pre-joint space
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    tinysol_instruments = ['bass tuba', 'french horn', 'trombone', 'trumpet', 'accordion', 'contrabass', 'violin', 'viola', 'cello', 'bassoon', 'clarinet', 'flute', 'oboe', 'alto saxophone']
    doktorski_sim = np.load('tinysol_instrument_similarity.npy')

    triplets = []
    triplets_indexes = []

    for anchor_index in range(len(tinysol_instruments)):
        for positive_index in range(anchor_index + 1, len(tinysol_instruments)):
            for negative_index in range(anchor_index + 1, len(tinysol_instruments)):
                if negative_index != positive_index:
                    if doktorski_sim[anchor_index,positive_index] > -1:
                        if doktorski_sim[anchor_index, positive_index] > doktorski_sim[anchor_index, negative_index]:
                            triplets.append([tinysol_instruments[anchor_index], tinysol_instruments[positive_index], tinysol_instruments[negative_index]])
                            triplets_indexes.append([anchor_index, positive_index, negative_index])

    for model in MODELS:
        text_embeddings = np.load(f'model_evaluation/{model}/text/pre_joint/muscall_prompt.npy')
        similarity = cosine_similarity(text_embeddings, text_embeddings)

        accuracy = 0
        for triplet in triplets_indexes:
            if similarity[triplet[0], triplet[1]] > similarity[triplet[0], triplet[2]]:
                accuracy += 1

        accuracy /= len(triplets_indexes)
        print(f'{model}: {accuracy}')

elif args.experiment == 'generate_figures':
    # Histogram of similarities and violin plots of similarities for muscall prompt
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']

    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')

    plt.figure(figsize=(10, 6))
    bins = np.arange(-1, 1.05, 0.05) 

    colors = ['blue', 'orange', 'red']

    for model in MODELS:
        audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
        prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/muscall_prompt.npy')
        gt_labels = groundtruth.iloc[:, 1:].columns.to_list()

        similarity = cosine_similarity(audio_embeddings, prompt_embeddings)

        plt.hist(similarity.ravel(), bins=bins, color=colors[MODELS.index(model)], alpha=0.5, label = model, density = True)
        sns.kdeplot(similarity.ravel(), color=colors[MODELS.index(model)], linewidth=2, linestyle='--')

    plt.title('Histogram of cosine similarity between audio and instrument labels in TinySOL')
    plt.xlabel('Similarity')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'evaluation_output/figures/muscall_prompt_histogram_of_similarities.png')
    plt.close()


    # positive versus negative histogram
    plt.figure(figsize=(6, 4))
    bins = np.arange(-0.5, 0.7, 0.05)

    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    model_names = ['Music CLAP', 'Music/Speech CLAP', 'MusCALL']
    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')

    colors = ['red', 'green']

    for i, model in enumerate(MODELS, start=1):

        # Load embeddings
        audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
        prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/muscall_prompt.npy')
        
        # Calculate cosine similarity
        similarity = cosine_similarity(audio_embeddings, prompt_embeddings)

        gt_labels = groundtruth.iloc[:, 1:].columns.to_list()
        positive_similarities = []
        negative_similarities = []
        for label in gt_labels:
            indexes_of_positive = groundtruth[groundtruth[label] == 1].index.to_list()
            positive_similarities.append(similarity[indexes_of_positive, gt_labels.index(label)])
            indexes_of_negative = groundtruth[groundtruth[label] == 0].index.to_list()
            negative_similarities.append(similarity[indexes_of_negative, gt_labels.index(label)])

        positive_similarities = np.concatenate(positive_similarities)
        negative_similarities = np.concatenate(negative_similarities)
        
        # Create subplot
        plt.subplot(len(MODELS), 1, i)
        
        # Plot positive histogram
        density_pos, _ = np.histogram(positive_similarities.ravel(), bins=bins, density=True)
        plt.bar(bins[:-1], density_pos / np.sum(density_pos), width=np.diff(bins), color=colors[1], alpha=0.8, label=f'positive')
        
        # Plot negative histogram
        density_neg, _ = np.histogram(negative_similarities.ravel(), bins=bins, density=True)
        plt.bar(bins[:-1], density_neg / np.sum(density_neg), width=np.diff(bins), color=colors[0], alpha=0.8, label=f'negative')

        plt.title(f'{model_names[i-1]}')
        plt.subplots_adjust(hspace = 1.1)
        plt.ylabel('')
        plt.grid(True)

        plt.yticks(np.arange(0, (density_pos / np.sum(density_pos)).max(), 0.1))
        
        # Add x-label to the last subplot
        if i == len(MODELS):
            plt.xlabel('Cosine Similarity', fontsize=13)
        elif i == 1:
            plt.legend()

    plt.figtext(0.01, 0.5, 'Probability', va='center', rotation='vertical', fontsize = 13)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.13)
    plt.savefig(f'evaluation_output/figures/muscall_prompt_histogram_of_positive_v_negative_similarities_subplot.png')
    plt.close()


    # Audio only positive vs negative histogram
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    model_names = ['Music CLAP', 'Music/Speech CLAP', 'MusCALL']
    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')

    plt.figure(figsize=(6, 4))
    bins = np.arange(0, 1.05, 0.05)

    colors = ['red', 'green']

    for space_type in ['joint_space']:
        for i, model in enumerate(MODELS):
            audio_embeddings = np.load(f'model_evaluation/{model}/audio/{space_type}/tinysol.npy')
            gt_labels = groundtruth.iloc[:, 1:].columns.to_list()

            label_embeddings = []

            for label in gt_labels:
                indexes = groundtruth[groundtruth[label] == 1].index.to_list()

                mean_embedding = np.mean(audio_embeddings[indexes, :], axis = 0)
                mean_embedding /= np.linalg.norm(mean_embedding)

                label_embeddings.append(mean_embedding)

            label_embeddings = np.stack(label_embeddings, axis = 0)

            similarity = cosine_similarity(audio_embeddings, label_embeddings)

            positive_similarities = []
            negative_similarities = []
            for label in gt_labels:
                indexes_of_positive = groundtruth[groundtruth[label] == 1].index.to_list()
                positive_similarities.append(similarity[indexes_of_positive, gt_labels.index(label)])
                indexes_of_negative = groundtruth[groundtruth[label] == 0].index.to_list()
                negative_similarities.append(similarity[indexes_of_negative, gt_labels.index(label)])
            
            positive_similarities = np.concatenate(positive_similarities)
            negative_similarities = np.concatenate(negative_similarities)
            
            # Create subplot
            plt.subplot(len(MODELS), 1, i+1)
            
            # Plot positive histogram
            density_pos, _ = np.histogram(positive_similarities.ravel(), bins=bins, density=True)
            plt.bar(bins[:-1], density_pos / np.sum(density_pos), width=np.diff(bins), color=colors[1], alpha=0.8, label=f'positive')
            
            # Plot negative histogram
            density_neg, _ = np.histogram(negative_similarities.ravel(), bins=bins, density=True)
            plt.bar(bins[:-1], density_neg / np.sum(density_neg), width=np.diff(bins), color=colors[0], alpha=0.8, label=f'negative')

            plt.title(f'{model_names[i]}')
            plt.subplots_adjust(hspace = 1.1)
            plt.ylabel('')
            plt.grid(True)

            plt.yticks(np.arange(0, (density_pos / np.sum(density_pos)).max(), 0.2))
            
            # Add x-label to the last subplot
            if i == len(MODELS)-1:
                plt.xlabel('Cosine Similarity', fontsize=13)

        plt.figtext(0.01, 0.5, 'Probability', va='center', rotation='vertical', fontsize = 13)
        plt.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.13)

        plt.savefig(f'evaluation_output/figures/audio_only_positive_negative_{space_type}.png')
        plt.close()

    # top-2 confidence
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']
    model_names = ['Music CLAP', 'Music/Speech CLAP', 'MusCALL']
    plt.figure(figsize=(6, 4))
    bins = np.arange(0.0, 0.3, 0.01) 

    colors = ['red', 'green']

    for i, model in enumerate(MODELS, start=1):

        # Load embeddings
        audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
        prompt_embeddings = np.load(f'model_evaluation/{model}/text/joint_space/muscall_prompt.npy')
        
        # Calculate cosine similarity
        similarity = cosine_similarity(audio_embeddings, prompt_embeddings)
        similarity = np.sort(similarity)

        top_2_difference = similarity[:, -1] - similarity[:, -2]

        
        # Create subplot
        plt.subplot(len(MODELS), 1, i)

        density_pos, _ = np.histogram(top_2_difference.ravel(), bins=bins, density=True)
        plt.bar(bins[:-1], density_pos / np.sum(density_pos), width=np.diff(bins), color=colors[1], alpha=0.8, label=f'positive')
        # plt.hist(top_2_difference.ravel(), bins=bins, color=colors[1], alpha=0.8, density=True, stacked=True)
        # sns.kdeplot(top_2_difference.ravel(), color=colors[1], linewidth=2, linestyle='--')

        plt.title(f'{model_names[i-1]}')
        # plt.xlabel('Top-2 class similarity difference')
        # plt.ylabel('Probability')
        plt.ylabel('')

        plt.subplots_adjust(hspace = 1.1)
        plt.grid(True)

        # Add xlabel to the second subplot
        if i == 3:
            plt.xlabel('Cosine similarity difference', fontsize=13)

    plt.figtext(0.01, 0.5, 'Probability', va='center', rotation='vertical', fontsize=13)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.13)

    plt.savefig(f'evaluation_output/figures/muscall_prompt_histogram_of_positive_v_negative_similarities_subplot.png')
    plt.close()


    # Histogram of similarities and violin plots of similarities for audio case
    MODELS = ['music_laion_clap', 'music_speech_laion_clap', 'muscall']

    groundtruth = pd.read_csv('tinysol_groundtruth_reordered.csv')

    for model in MODELS:
        audio_embeddings = np.load(f'model_evaluation/{model}/audio/joint_space/tinysol.npy')
        gt_labels = groundtruth.iloc[:, 1:].columns.to_list()

        label_embeddings = []

        for label in gt_labels:
            indexes = groundtruth[groundtruth[label] == 1].index.to_list()

            mean_embedding = np.mean(audio_embeddings[indexes, :], axis = 0)
            mean_embedding /= np.linalg.norm(mean_embedding)

            label_embeddings.append(mean_embedding)

        label_embeddings = np.stack(label_embeddings, axis = 0)

        positive_similarities = []
        negative_similarities = []

        for label in gt_labels:
            indexes = groundtruth[groundtruth[label] == 1].index.to_list()

            mean_embedding = np.mean(audio_embeddings[indexes, :], axis = 0)
            mean_embedding /= np.linalg.norm(mean_embedding)

            label_embeddings.append(mean_embedding)

