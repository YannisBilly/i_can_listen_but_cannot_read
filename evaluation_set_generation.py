import csv
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog = 'I can listen but cannot read',
                                 description = 'Oficial implementatino of paper accepted at ISMIR 2024')

parser.add_argument('-et', '--evaluation_type', help = 'Which evaluation set to generate. Use "tinysol" for tinysol only labels or "all" to use Doktorski ontology.',
                    default='all')

args = parser.parse_args()

# Doktorski ontology
ontology = {
    "instrument": {
        'stringed': {
            'bowed': ['violin', 'viola', 'violoncello', 'contrabass'],
            'plucked': ['guitar', 'banjo', 'ukulele', 'harp', 'harpsichord'],
            'struck': ['hammered_dulcimer', 'piano']
        },
        'wind': {
            'pipe_aerophones': {
                'edge': {
                    'whistle_flutes': ['whistle', 'recorder', 'organ_flue_pipes'],
                    'true_flutes': ['jug', 'panpipes', 'flute', 'piccolo']
                },
                'reed_pipe': {
                    'single_reeds': ['clarinet', 'saxophone', 'single_reed_bagpipe'],
                    'double_reeds': ['oboe', 'bassoon', 'double_reed_bagpipe']
                },
                'brass': {
                    'without_valves': ['conch_shell', 'animal_horn_shofar', 'didjeridu', 'bugle', 'trombone'],
                    'with_valves': ['trumpet', 'cornet', 'french_horn', 'euphonium', 'tuba']
                }
            },
            'free_aerophones': {
                'beating_reed': {
                    'single_reed': ['hautbois', 'fagotto', 'chalumeau', 'krummhorn', 'clairon', 'trompette', 'trompette en chamade'], #removing trombone and tuba from here
                    'double_reed': ['voice']
                },
                'free_reed': {
                    'unframed_reed': {
                        'wind_blown': ['bull_roarer', 'aeolian_harp'],
                        'mouth_blown': ['leaf_instrument'],
                        'mouth_blown_and_plucked': ['jew_harp']
                    },
                    'framed_reed': {
                        'mouth_blown': ['sheng', 'sho', 'khaen', 'harmonica'],
                        'hand_blown': ['concertina', 'bandoneon', 'bayan', 'accordion', 'indian_harmonium'],
                        'foot_blown': ['harmonium', 'reed_organ', 'pedal_concertina'],
                        'mechanically_blown': ['barrel_organ', 'orchestrion', 'pedal_reed_organ', 'electric_chord_organ']
                    }
                }
            }
        },
        'percussion': {
            'idiophones': {
                'pitched': {
                    'struck': ['triangle', 'bell', 'gong', 'cymbal', 'xylophone', 'marimba', 'celesta'],
                    'rubbed': ['prayer_bowls', 'glass_harmonica'],
                    'plucked': ['music_box', 'kalimba', 'mbira']
                },
                'unpitched': {
                    'struck': ['slit_drum', 'castanets'],
                    'shaken': ['rattle', 'jingles']
                }
            },
            'membranophone': {
                'determinate_pitch': {
                    'struck': ['timpani', 'roto_toms']
                    }
                },
                'indeterminate_pitch': {
                    'struck': ['snare_drum', 'bass_drum', 'bongos', 'congas', 'tambourine'],
                    'rubbed': ['friction_drum'],
                    'blown': ['kazoo']
                }
            }
        },
        'electrophone': {
            'electroacoustic': [ 'electric_guitar', 'electric_bass', 'fender_rhodes_electric_piano', 'electric_violin'],
            'electronic': {
                'electromagnetic': ['theremin', 'ondes_martenot', 'electric_organ', 'synthesizer'],
                'digital': ['midi_keyboard', 'midi_wind_controller', 'midi_drum_machine', 'midi_guitar', 'midi_accordion']
            }
        }
    }

def flatten_ontology(ontology):
    nodes = []
    for key, value in ontology.items():
        if isinstance(value, dict):
            nodes.append(key)
            nodes.extend(flatten_ontology(value))
        elif isinstance(value, list):
            nodes.append(key)
            nodes.extend([item for item in value])
    return nodes

from collections import deque

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

def build_tree(node, subtree):
    for key, value in subtree.items():
        child = TreeNode(key)
        if isinstance(value, dict):
            build_tree(child, value)
        elif isinstance(value, list):
            for item in value:
                child.children.append(TreeNode(item))
        node.children.append(child)

def shortest_distance(root, node1, node2):
    if root is None or node1 is None or node2 is None:
        return -1

    def find_path(node, target):
        if node is None:
            return None
        if node.name == target:
            return [node.name]
        for child in node.children:
            path = find_path(child, target)
            if path:
                return [node.name] + path
        return None

    path1 = find_path(root, node1)
    path2 = find_path(root, node2)

    if not path1 or not path2:
        return -1

    i = 0
    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
        i += 1

    if i == 2:
        return -1

    return len(path1) + len(path2) - 2 * i

def find_distance_to_root(root, node_name):
    def find_path_to_root(node, target, path):
        if node is None:
            return False
        path.append(node.name)
        if node.name == target:
            return True
        for child in node.children:
            if find_path_to_root(child, target, path):
                return True
        path.pop()
        return False

    path_to_root = []
    find_path_to_root(root, node_name, path_to_root)
    return len(path_to_root) - 1  # Subtract 1 to exclude the root itself from the distance count


def get_leaf_nodes(ontology):
    def _get_leaf_nodes_helper(subtree):
        leaf_nodes = []
        for key, value in subtree.items():
            if isinstance(value, dict):
                leaf_nodes.extend(_get_leaf_nodes_helper(value))
            elif isinstance(value, list):
                leaf_nodes.extend([item for item in value])
        if not isinstance(subtree, list) and not isinstance(subtree, dict):
            leaf_nodes.append('')
        return leaf_nodes
    
    return _get_leaf_nodes_helper(ontology)

tree = TreeNode('instrument_ontology')
build_tree(tree, ontology)

if args.evaluation_type == 'all':
    triplets = []

    ontology_list = flatten_ontology(ontology)[1:]

    leaf_nodes = get_leaf_nodes(ontology)

    with open('doktorski_triplets.csv', 'w') as file:
        writer = csv.writer(file)

        writer.writerow(['anchor', 'positive', 'negative'])
        for term_1_counter, term_1 in enumerate(tqdm(ontology_list)):
            for term_2 in ontology_list[term_1_counter+1:]:
                if not(term_1 == term_2):
                    pos_term_distance = shortest_distance(tree, term_1, term_2)

                    if pos_term_distance > 0:
                        for term_3 in ontology_list:
                            if not(term_3 == term_1) and not(term_3 == term_2):
                                if shortest_distance(tree, term_1, term_3) < pos_term_distance:
                                    if not(term_1 in leaf_nodes) and not('instruments' in term_1):
                                        term_1 = term_1 + " instruments"
                                    if not(term_2 in leaf_nodes) and not('instruments' in term_2):
                                        term_2 = term_2 + " instruments"
                                    if not(term_3 in leaf_nodes) and not('instruments' in term_3):
                                        term_3 = term_3 + " instruments"
                                    triplets.append([term_1, term_2, term_3])
                                    writer.writerow([term_1, term_2, term_3])
elif args.evaluation_type:
    tinysol_instruments = ['tuba', 'french_horn' , 'trombone' , 'trumpet' , 'accordion' , 'contrabass' , 'violin' , 'viola' , 'violoncello' , 'bassoon' , 'clarinet' , 'flute' , 'oboe' , 'saxophone']

    doktorski_similarity = np.zeros((len(tinysol_instruments), len(tinysol_instruments)))
                                    
    for i in range(len(tinysol_instruments)):
        for j in range(len(tinysol_instruments)):
            doktorski_similarity[i,j] = shortest_distance(tree, tinysol_instruments[i], tinysol_instruments[j])

    np.save('tinysol_instrument_similarity.npy', doktorski_similarity)