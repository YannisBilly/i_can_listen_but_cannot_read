import torch
import laion_clap
import librosa as lr
import numpy as np
import lorem
from transformers import RobertaTokenizer
import argparse
import os
import pandas as pd
from tqdm import tqdm
import torchaudio as ta

class multimodal_system():
    def __init__(self, model, model_type = 'clap', device = 'cuda:0'):
        self.model = model
        self.type = model_type
        self.device = device
        
        if model_type == 'clap':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif model_type == 'muscall':
            pass

    def generate_joint_space_audio_embeddings(self, audio):
        if self.type == 'clap':
            return self.model.get_audio_embedding_from_data(x = audio, use_tensor = True)

        elif self.type == 'muscall':
            pass

        else:
            print('Not a valid model')

    def generate_joint_space_text_embeddings(self, text):
        if self.type == 'clap':
            if len(text) == 1:
                text += text
                embed = self.model.get_text_embedding(text)
                return embed[0:1, :]
            else:
                return self.model.get_text_embedding(text)

        elif self.type == 'muscall':
            pass

        else:
            print('Not a valid model')

    def generate_pre_projection_audio_embeddings(self, audio):
        audio = audio[0, :]

        if self.type == 'clap':
            audio_len = len(audio)

            k = self.model.model.audio_cfg.clip_samples // audio_len
            if k > 1:
                audio = audio.repeat(k)
                audio_len = len(audio)

            if audio_len > self.model.model.audio_cfg.clip_samples:
                audio = torch.tensor(audio, device = self.device)
                audio_input = [
                    audio[pos : pos + self.model.model.audio_cfg.clip_samples].clone()
                
                    for pos in range(
                        0, audio_len - self.model.model.audio_cfg.clip_samples, audio_len
                    )
                ]
                
                audio_input.append(audio[-self.model.model.audio_cfg.clip_samples :].clone())
                audio_input = torch.stack(audio_input)
                
                temp_dict = {}
                temp_dict['waveform'] = audio_input
                embed = self.model.model.encode_audio(temp_dict, device = self.device)['embedding']
                embed = torch.mean(embed, axis = 0).unsqueeze(0)
                
            else:
                temp_dict = {}
                temp_dict['waveform'] = torch.tensor(np.expand_dims(audio, axis = 0), device = self.device, dtype=torch.float)
                embed = self.model.model.encode_audio(temp_dict, device = self.device)['embedding']

            return embed

        elif self.type == 'muscall':
            pass

        else:
            print('Not a valid model')

    def generate_pre_projection_text_embeddings(self, text):
        if self.type == 'clap':
            text = self.tokenizer(text, padding = 'max_length', return_tensors = "pt", truncation = True, max_length = 77)

            return model.model.text_branch(
                input_ids=text["input_ids"].to(device=self.device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=self.device, non_blocking=True
                ),
            )["pooler_output"].detach().cpu().numpy()

        elif self.type == 'muscall':
            pass

        else:
            print('Not a valid model')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of two-tower multimodal systems")

    parser.add_argument(
        "--device_num",
        type=str,
        default="0",
    )

    parser.add_argument(
        '--model',
        type=str,
        default = 'music_laion_clap'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default = 'TinySOL'
    )

    args = parser.parse_args()

    return args

# quantization
def int16_to_float32(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16(x):
    x = torch.clamp(x, min=torch.tensor(-1.), max=torch.tensor(1.))
    return (x * 32767.).type(torch.int16)


parser = parse_args()

tinysol_instruments = ['bass tuba', 'french horn', 'trombone', 'trumpet in c',
                        'accordion', 'contrabass', 'violin', 'viola',
                        'violoncello', 'bassoon', 'clarinet in b-flat',
                        'flute', 'oboe', 'alto saxophone']

definitions = ['a bass tuba is a large brass instrument with a deep, rich sound that typically plays in the lower registers of the musical ensemble. it is distinguished by its large size and conical shape, which allows for a resonant and powerful sound production. the bass tuba is an integral part of the brass section, providing a strong foundation and grounding the ensemble with its low tones. its distinctive sound quality adds depth and richness to the overall musical texture, making it a crucial component in many symphonic compositions and brass band arrangements.',
               "the french horn is a brass instrument known for its rich, warm tone quality and distinctive, mellow sound. it is commonly found in orchestras and concert bands, adding depth and color to musical compositions. the french horn is unique in its shape and design, featuring a coiled tube with a flared bell at the end. players produce sound by buzzing their lips into a mouthpiece and manipulating the valves to change the pitch. the french horn's range extends from the low, resonant notes to the bright, soaring highs, making it a versatile instrument with a wide tonal variety. its timbre blends well with other brass and woodwind instruments, creating a harmonious and expressive musical texture.",
               "the trombone is a brass instrument with a unique sliding mechanism that allows the player to change the pitch smoothly. unlike other instruments, the trombone does not have discrete keys or valves to change notes; instead, the player uses a slide to adjust the length of the tubing, creating a smooth glissando effect. this distinctive feature gives the trombone a rich, warm sound that can be both commanding and lyrical. the trombone is known for its versatility in different musical genres, from classical orchestras to jazz ensembles, and its robust sound can add depth and gravitas to any musical piece.",
               'a trumpet in c refers to a specific type of trumpet that is built to sound in the key of c. unlike larger wind instruments such as bass tuba or contrabass, a trumpet in c has a higher pitch and produces brighter, more piercing tones. it is smaller and more compact than a trombone or french horn, allowing it to reach higher notes more easily. the sound of a trumpet in c is distinct from the warm, rich tones of a violoncello or the deep, resonant notes of a bassoon. its sharp and clear sound cuts through a musical ensemble, making it a versatile and prominent instrument in various genres of music.',
               "the accordion is a unique musical instrument that produces sound by bellows and by pressing keys or buttons. it is distinct from others in its ability to create a rich, harmonious sound with a wide range of timbres. the accordion's signature feature is the ability to play chords, melody, and bass simultaneously, offering a full and versatile sound. its expressive nature and dynamic range make it a compelling instrument in various musical genres. moreover, the accordion's portable design and intuitive playability add to its appeal for musicians looking to enhance their sonic palette.",
               "contrabass refers to the lowest-pitched instrument in an ensemble, known for producing deep and resonant sounds that provide a strong foundation in the lower registers. it is significantly deeper in pitch compared to other instruments, creating a rich and full-bodied tone that adds depth and gravitas to musical compositions. the contrabass typically has a larger size and longer strings or tubing, allowing it to produce the lowest notes with clarity and power. its distinct tone quality and sonic presence make it an essential component in orchestral and ensemble settings, adding warmth and weight to the overall sound.",
               "the violin is a string instrument known for its high-pitched, bright sound. its compact size and the way it is played with a bow set it apart from the larger brass instruments, such as the bass tuba, french horn, trombone, and trumpet in c. the violin's agility in producing rich, melodic tones distinguishes it from the harmonic capabilities of the accordion and the deep, resonant tone of the contrabass. additionally, the violin's smaller size and higher pitch differentiate it from the lower-range string instruments like the viola, violoncello, and bassoon. its sound quality contrasts with the clarinet in b flat's reedy tone and the smooth, airy timbre of the flute. lastly, the violin's expressive capabilities set it apart from the warm, wooden tone of the oboe and the soulful, versatile sound of the alto saxophone. the violin's unique sound and playing technique make it a standout in the world of musical instruments.",
               "the viola is a string instrument that is larger than a violin and smaller than a violoncello. it has a warm, rich tone that sits in the middle range of the string family, producing a mellow sound that is both melodic and supportive in ensemble settings. the viola is played with a bow made of horsehair and can also be plucked to create a variety of textures. its unique size and tuning give it a distinct sound that complements and enhances the overall harmonies of a musical piece, making it a versatile and essential instrument in orchestras and chamber music ensembles.",
               "the term violoncello refers to a large string instrument that is held between the legs and played with a bow. it is larger than a violin and viola, but smaller than a double bass. the sound produced by the violoncello is warm, rich, and resonant, lending itself to both melodic lines and supportive harmonies in various musical settings.the violoncello's range spans from deep, sonorous low notes to sweet, singing high notes, providing a versatile sonic palette for composers and performers. its timbre is distinct, offering a lush and expressive quality that can evoke a wide range of emotions.when compared to brass and woodwind instruments, the violoncello's sound is more continuous and fluid, allowing for long, sustained tones and expressive phrasing. its bowed nature also contributes to the smooth, connected quality of its sound, making it well-suited for lyrical passages and emotive melodies.in ensemble settings, the violoncello often serves as a crucial link between the lower and upper registers, providing a solid foundation for harmony while also adding depth and richness to the overall sound. its ability to blend with other instruments or stand out in solo passages makes it a versatile and essential instrument in classical, jazz, and contemporary music.",
               "the bassoon is a woodwind instrument with a double reed that produces a distinctive, deep, and rich tone. unlike many brass instruments that rely on a mouthpiece and valves to produce sound, the bassoon uses a complex system of keys, holes, and reeds to create its unique timbre. it has a wide range, spanning from the low, resonant notes to the high, expressive registers. the bassoon's sound is often described as sonorous, full-bodied, and reedy, making it a crucial member of the woodwind family in classical orchestras and chamber ensembles. it has a unique blend of agility and lyricism, capable of both nimble runs and poignant melodies. the bassoon's unmistakable sound adds depth and color to any musical composition, providing a solid foundation in the lower register while also capable of soaring in the upper range with haunting beauty.",
               "the term clarinet in b flat refers to a specific type of clarinet that is tuned to the key of b flat. this means that when a player reads a note on the musical staff, the sound produced will be a different pitch compared to the instruments listed in brackets. the clarinet in b flat has a warm, mellow sound with a distinct timbre that is often described as rich and versatile. its range spans over three octaves and it is known for its ability to blend well with other instrument families in ensemble settings. the clarinet in b flat is a popular choice in various genres of music, from classical to jazz and beyond, and its unique sound adds depth and texture to any musical composition.",
               "the flute is a woodwind instrument known for its light, airy sound quality and high pitch range. unlike the other instruments listed, the flute produces sound by blowing across a mouthpiece hole rather than using a reed or a brass mouthpiece. it is often made of metal or wood and consists of a long tube with keys to change pitch. the flute's tone is smooth, pure, and delicate, making it a versatile instrument capable of expressing a wide range of emotions. its sound is characterized by its ability to effortlessly glide between notes with a shimmering quality, making it a popular choice in orchestral, chamber, and solo music.",
               "the oboe is a woodwind instrument known for its distinctive timbre and piercing sound. it is smaller in size compared to larger brass instruments like the bass tuba and trombone. unlike the trumpet in c or accordion, the oboe has a double reed that produces a unique and focused tone. it is also much higher in pitch compared to the deep tones of the contrabass. the oboe's sound stands out in orchestral settings, adding a bright and expressive quality that can be compared to the string instruments such as violin, viola, and violoncello. moreover, it has a more nasal and penetrating sound compared to the mellow tones of the bassoon or the smooth sound of the clarinet in b flat. the oboe's sound is distinct from the airy and light tones of the flute and alto saxophone.",
               "the alto saxophone is a woodwind instrument that belongs to the saxophone family. it is slightly smaller than the tenor saxophone and larger than the soprano saxophone. the sound of the alto saxophone is characterized by its bright and expressive tone, sitting in the mid-range of the saxophone family. it has a curved neck and a distinctive s-shaped body that gives it a unique visual appeal. the alto saxophone is commonly used in jazz, classical, and contemporary music genres, where its versatility and agility allow for melodic and expressive playing. with a rich and resonant sound, the alto saxophone can be both powerful and lyrical, making it a popular choice for soloists and ensemble players alike."
]

definitions_without_labels = []

for definition, label in zip(definitions, tinysol_instruments):
    words_in_label = label.split(' ')

    for word in words_in_label:
        definition = definition.replace(word, '')

    definition = definition.replace('   ', ' ')

    definitions_without_labels.append(definition)

label_prompt_titles = ["muscall_prompt", 'label_random_context', 'negative_muscall_prompts', 'this_is_a_recording_of',
                       'solo_musical_instrument_sound_of', 'sound_of_musical_instrument', 'definition', 'definition_no_label_word']
label_prompts = [[f'A {x} track' for x in tinysol_instruments],
           [f'{x} {lorem.sentence().lower()}.' for x in tinysol_instruments],
           [f'A no {x} track' for x in tinysol_instruments],
           [f'This is a recording of {x}' for x in tinysol_instruments],
           [f'Solo musical instrument sound of {x}' for x in tinysol_instruments],
           [f'Sound of {x} musical instrument' for x in tinysol_instruments],
           definitions,
           definitions_without_labels
]

# instrument family
instrument_families = ['key', 'brass', 'woodwind', 'string']

instrument_family_titles = ['muscall_prompt_instruments', 'muscall_prompt_and_instruments', 'muscall_prompt_and_musical_instruments']
instrument_prompts = [[f'A {x} track' for x in instrument_families],
                      [f'A {x} instruments track' for x in instrument_families],
                      [f'A track of {x} musical instruments' for x in instrument_families]
]

MODELS = ['audioset_laion_clap', 'music_laion_clap', 'music_speech_laion_clap']

device = f'cuda:{parser.device_num}'

if parser.model == 'music_laion_clap':
    model = laion_clap.CLAP_Module(enable_fusion = False,  amodel= 'HTSAT-base')
    model.load_ckpt('checkpoints/music_audioset_epoch_15_esc_90.14.pt')
    model.to(device)
    model.eval()
elif parser.model == 'music_speech_laion_clap':
    model = laion_clap.CLAP_Module(enable_fusion = False,  amodel= 'HTSAT-base')
    model.load_ckpt('checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt')
    model.to(device)
    model.eval()

model_wrapper = multimodal_system(model, device=device)

path_to_save = 'model_evaluation/'
if not(os.path.exists(path_to_save)):
    os.mkdir(path_to_save)

path_to_save = os.path.join(path_to_save, parser.model)
if not(os.path.exists(path_to_save)):
    os.mkdir(path_to_save)
    os.mkdir(os.path.join(path_to_save, 'text'))
    os.mkdir(os.path.join(path_to_save, 'audio'))

# prompt embeddings generation in joint space
details_for_save = os.path.join('text', 'joint_space')
if not(os.path.exists(os.path.join(path_to_save, details_for_save))):
    os.mkdir(os.path.join(path_to_save, details_for_save))

for experiment_type, prompts in zip(label_prompt_titles, label_prompts):
    embeddings = model_wrapper.generate_joint_space_text_embeddings(prompts)
    np.save(os.path.join(path_to_save, details_for_save, experiment_type + ".npy"), embeddings)

for experiment_type, prompts in zip(instrument_family_titles, instrument_prompts):
    embeddings = model_wrapper.generate_joint_space_text_embeddings(prompts)
    np.save(os.path.join(path_to_save, details_for_save, experiment_type + ".npy"), embeddings)

# prompt emeddings generation in pre-joint space
details_for_save = os.path.join('text', 'pre_joint')
if not(os.path.exists(os.path.join(path_to_save, details_for_save))):
    os.mkdir(os.path.join(path_to_save, details_for_save))

for experiment_type, prompts in zip(label_prompt_titles, label_prompts):
    embeddings = model_wrapper.generate_pre_projection_text_embeddings(prompts)
    np.save(os.path.join(path_to_save, details_for_save, experiment_type + ".npy"), embeddings)

for experiment_type, prompts in zip(instrument_family_titles, instrument_prompts):
    embeddings = model_wrapper.generate_pre_projection_text_embeddings(prompts)
    np.save(os.path.join(path_to_save, details_for_save, experiment_type + ".npy"), embeddings)

path_to_joint_audio = os.path.join(path_to_save, 'audio', 'joint_space')
path_to_pre_joint_audio = os.path.join(path_to_save, 'audio', 'pre_joint')

# audio embeddings generation
if not(os.path.exists(path_to_joint_audio)):
    os.mkdir(path_to_joint_audio)
if not(os.path.exists(path_to_pre_joint_audio)):
    os.mkdir(path_to_pre_joint_audio)

groundtruth = pd.read_csv('tinysol_groundtruth.csv')
song_names = groundtruth['song'].to_list()

joint_embeddings = []
pre_joint_embeddings = []

resample = ta.transforms.Resample(44100, 48000)

for song in tqdm(song_names):
    audio, _ = ta.load(os.path.join('TinySOL', song + '.wav'))
    audio = resample(audio)

    audio = int16_to_float32(float32_to_int16(audio))

    embeddings = model_wrapper.generate_joint_space_audio_embeddings(audio)
    joint_embeddings.append(embeddings.detach().cpu().numpy())

    embeddings = model_wrapper.generate_pre_projection_audio_embeddings(audio)
    pre_joint_embeddings.append(embeddings.detach().cpu().numpy())

joint_embeddings = np.squeeze(np.stack(joint_embeddings, axis = 0))
np.save(os.path.join(path_to_joint_audio, 'tinysol.npy'), joint_embeddings)

pre_joint_embeddings = np.squeeze(np.stack(pre_joint_embeddings, axis = 0))
np.save(os.path.join(path_to_pre_joint_audio, 'tinysol.npy'), pre_joint_embeddings)
