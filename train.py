import json

from argparse import ArgumentParser
from model import *
import numpy as np
import time
import torch
import csv
import utils
import torch.utils.data
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    drive_path = '/content/drive/My Drive/Colab Notebooks/NLP_A4/'
else:
    drive_path = ''

colors = {'y': 1, 'o': 2, 'r': 3, 'g': 4, 'b': 5, 'p': 6}
numbers = {'1' : 'y', '2': 'o', '3': 'r', '4': 'g', '5': 'b', '6': 'p'}

def process_state(state_string):
    beaker_strings = state_string.split(' ')
    state = []
    for beaker_string in beaker_strings:
        chunks = beaker_string.split(':')
        if len(chunks) < 2: # should only occur with all utterances in the test set, except for the first of each interaction
            state.append([])
            state.append([])
            state.append([])
            state.append([])
            state.append([])
            state.append([])
            state.append([])
            continue
        height = 0
        beaker_state = []
        for c in chunks[1]:
            if c in colors:
                beaker_state.append(colors[c])
        state.append(beaker_state)
    return state

def process_data(filename):
    print("Processing file " + str(filename))
    with open(filename) as f:
        data = json.load(f)
    after_env = []
    pairs = []
    V_input = []
    input_length = []
    V_output = []
    output_length = []

    for i in range(len(data)):
        cur_lst  = data[i]["utterances"]
        initial_env = data[i]["initial_env"]
        for j in range(len(cur_lst)):
            cur_dict = cur_lst[j]
            cur_x = cur_dict["instruction"]
            cur_x = "SOS " + cur_x + " EOS"
            cur_y = cur_dict["actions"]
            adj_cur_y_col = []
            for l in cur_y:
                #adj_cur_y_col.append(" ".join(l.split()))
                adj_cur_y_col.append("_".join(l.split())) 
            adj_cur_y = " ".join(adj_cur_y_col)
            
            cur_y = adj_cur_y + " EOS"
        
            cur_after_env = cur_dict["after_env"]    
            V_input.extend(cur_x.split())
            input_length.append(len(cur_x.split()))
            output_length.append(len(cur_y.split()))
            V_output.extend(cur_y.split())
            after_env.append(cur_after_env)
            
            env = process_state(initial_env)
            pairs.append([cur_x,cur_y, env])
            
            #initial_env = cur_after_env
    vocab_input = set(V_input)
    vocab_output = set(V_output)
    return pairs,vocab_input,vocab_output,max(input_length), max(output_length)

def process_data2(filename):
    print("Processing file " + str(filename))
    with open(filename) as f:
        data = json.load(f)
    after_env = []
    pairs = []
    V_input = []
    input_length = []
    V_output = []
    output_length = []

    for i in range(len(data)):
        cur_lst  = data[i]["utterances"]
        initial_env = data[i]["initial_env"]
        for j in range(len(cur_lst)):
            cur_dict = cur_lst[j]
            cur_x = cur_dict["instruction"]
            cur_x = "SOS " + cur_x + " EOS"
            cur_y = cur_dict["actions"]
            adj_cur_y_col = []
            for l in cur_y:
                #adj_cur_y_col.append(" ".join(l.split()))
                adj_cur_y_col.append(" ".join(l.split())) 
            adj_cur_y = " , ".join(adj_cur_y_col)
            
            cur_y = adj_cur_y + " EOS"
        
            cur_after_env = cur_dict["after_env"]    
            V_input.extend(cur_x.split())
            input_length.append(len(cur_x.split()))
            output_length.append(len(cur_y.split()))
            V_output.extend(cur_y.split())
            after_env.append(cur_after_env)
            
            env = process_state(initial_env)
            pairs.append([cur_x,cur_y, env])
            
            #initial_env = cur_after_env
    vocab_input = set(V_input)
    vocab_output = set(V_output)
    return pairs,vocab_input,vocab_output,max(input_length), max(output_length)

def process_data_part4(filename):
    print("Processing file " + str(filename))
    with open(filename) as f:
        data = json.load(f)
    after_env = []
    pairs = []
    V_input = []
    input_length = []
    V_output = []
    output_length = []

    for i in range(len(data)):
        cur_lst  = data[i]["utterances"]
        initial_env = data[i]["initial_env"]
        prev_x = "-empty-"
        for j in range(len(cur_lst)):
            cur_dict = cur_lst[j]
            cur_x = cur_dict["instruction"]
            cur_x = "SOS " + cur_x + " EOS"
            cur_y = cur_dict["actions"]
            adj_cur_y_col = []
            for l in cur_y:
                #adj_cur_y_col.append(" ".join(l.split()))
                adj_cur_y_col.append(" ".join(l.split())) 
            adj_cur_y = " , ".join(adj_cur_y_col)
            
            cur_y = adj_cur_y + " EOS"
        
            cur_after_env = cur_dict["after_env"]    
            V_input.extend(cur_x.split())
            input_length.append(len(cur_x.split()))
            output_length.append(len(cur_y.split()))
            V_output.extend(cur_y.split())
            after_env.append(cur_after_env)
            
            env = process_state(initial_env)
            pairs.append([prev_x,cur_x,cur_y, env])
            
            prev_x = cur_x
            
            #initial_env = cur_after_env
    V_input.append("-empty-")
    vocab_input = set(V_input)
    vocab_output = set(V_output)
    return pairs,vocab_input,vocab_output,max(input_length), max(output_length)

def load_data(train_json, dev_json, test_json, split_actions = False):
    """Loads the data from the JSON files.

    You are welcome to create your own class storing the data in it; e.g., you
    could create AlchemyWorldStates for each example in your data and store it.

    Inputs:
        filename (str): Filename of a JSON encoded file containing the data. 

    Returns:
        examples
    """
    if split_actions:
        pairs_tr,vocab_input_tr,vocab_output_tr,max_input_length,max_output_length = process_data2(train_json)
        pairs_dev,vocab_input_dev,vocab_output_dev,_,_ = process_data2(dev_json)
        pairs_te,vocab_input_te,vocab_output_te,_,_ = process_data2(test_json)
    else:
        pairs_tr,vocab_input_tr,vocab_output_tr,max_input_length,max_output_length = process_data(train_json)
        pairs_dev,vocab_input_dev,vocab_output_dev,_,_ = process_data(dev_json)
        pairs_te,vocab_input_te,vocab_output_te,_,_ = process_data(test_json)
        
    print("Processing %s training instruction-action pairs" % len(pairs_tr))
    print("Processing %s dev instruction-action pairs" % len(pairs_dev))
    print("Processing %s test instruction-action pairs" % len(pairs_te))
    vocab_input = vocab_input_tr.union(vocab_input_dev, vocab_input_te)
    vocab_output = vocab_output_tr.union(vocab_output_dev, vocab_output_te)
    # index 0 is reserved for padding, index 1 is SOS, index2 is EOS
    vocab_input.remove('SOS')
    vocab_input.remove('EOS')
    vocab_output.remove('EOS')
    word_to_idx_input = {w: idx + 3 for idx, w in enumerate(vocab_input)}
    word_to_idx_input.update({'SOS':1, 'EOS':2})
    idx_to_word_input = {idx + 3: w for idx, w in enumerate(vocab_input)}
    idx_to_word_input.update({1: 'SOS', 2: 'EOS'})
    word_to_idx_output = {w: idx + 3 for idx, w in enumerate(vocab_output)}
    word_to_idx_output.update({'SOS':1, 'EOS':2})
    idx_to_word_output = {idx + 3: w for idx, w in enumerate(vocab_output)}
    idx_to_word_output.update({1: 'SOS', 2: 'EOS'})
    print("Maximum input instruction length (based on training set) is: ", max_input_length)
    print("Output vocab size is: ", len(vocab_output))
    
    max_beaker_height = 4
    
    tr_x_mtx = np.zeros((len(pairs_tr), max_input_length + (max_beaker_height * 7)))
    tr_y_mtx = np.zeros((len(pairs_tr), max_output_length))
    dev_x_mtx = np.zeros((len(pairs_dev), max_input_length + (max_beaker_height * 7)))
    te_x_mtx = np.zeros((len(pairs_te), max_input_length + (max_beaker_height * 7)))
    
    for i in range(len(pairs_tr)):
        x = pairs_tr[i][0]
        y = pairs_tr[i][1]
        words = x.split(" ")
        for j in range(len(words)):
            tr_x_mtx[i][j] = word_to_idx_input[words[j]] 
        words = y.split(" ")
        for j in range(len(words)):
            tr_y_mtx[i][j] = word_to_idx_output[words[j]]
        initial_env = pairs_tr[i][2]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    tr_x_mtx[i][max_input_length + (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue
            
        
    for i in range(len(pairs_dev)):
        x = pairs_dev[i][0]
        y = pairs_dev[i][1]
        words = x.split(" ")
        for j in range(min(len(words), max_input_length)):
            dev_x_mtx[i][j] = word_to_idx_input[words[j]]
        initial_env = pairs_dev[i][2]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    dev_x_mtx[i][max_input_length + (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue
            
    for i in range(len(pairs_te)): 
        x = pairs_te[i][0]
        y = pairs_te[i][1]
        words = x.split(" ")
        for j in range(min(len(words), max_input_length)):
            te_x_mtx[i][j] = word_to_idx_input[words[j]]  
        initial_env = pairs_te[i][2]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    te_x_mtx[i][max_input_length + (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue    
                               
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(tr_x_mtx.astype(int)), torch.from_numpy(tr_y_mtx.astype(int)))
    dev_data = torch.utils.data.TensorDataset(torch.from_numpy(dev_x_mtx.astype(int)), torch.zeros((dev_x_mtx.shape[0], 1)))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(te_x_mtx.astype(int)), torch.zeros((te_x_mtx.shape[0], 1)))
    return train_data, dev_data, test_data, idx_to_word_input, idx_to_word_output


def load_data_part4(train_json, dev_json, test_json):
    """Loads the data from the JSON files.

    You are welcome to create your own class storing the data in it; e.g., you
    could create AlchemyWorldStates for each example in your data and store it.

    Inputs:
        filename (str): Filename of a JSON encoded file containing the data. 

    Returns:
        examples
    """
    pairs_tr,vocab_input_tr,vocab_output_tr,max_input_length,max_output_length = process_data_part4(train_json)
    pairs_dev,vocab_input_dev,vocab_output_dev,_,_ = process_data_part4(dev_json)
    pairs_te,vocab_input_te,vocab_output_te,_,_ = process_data_part4(test_json)
      
        
    print("Processing %s training instruction-action pairs" % len(pairs_tr))
    print("Processing %s dev instruction-action pairs" % len(pairs_dev))
    print("Processing %s test instruction-action pairs" % len(pairs_te))
    vocab_input = vocab_input_tr.union(vocab_input_dev, vocab_input_te)
    vocab_output = vocab_output_tr.union(vocab_output_dev, vocab_output_te)
    # index 0 is reserved for padding, index 1 is SOS, index2 is EOS
    vocab_input.remove('SOS')
    vocab_input.remove('EOS')
    vocab_output.remove('EOS')
    word_to_idx_input = {w: idx + 3 for idx, w in enumerate(vocab_input)}
    word_to_idx_input.update({'SOS':1, 'EOS':2})
    idx_to_word_input = {idx + 3: w for idx, w in enumerate(vocab_input)}
    idx_to_word_input.update({1: 'SOS', 2: 'EOS'})
    word_to_idx_output = {w: idx + 3 for idx, w in enumerate(vocab_output)}
    word_to_idx_output.update({'SOS':1, 'EOS':2})
    idx_to_word_output = {idx + 3: w for idx, w in enumerate(vocab_output)}
    idx_to_word_output.update({1: 'SOS', 2: 'EOS'})
    print("Maximum input instruction length (based on training set) is: ", max_input_length)
    print("Output vocab size is: ", len(vocab_output))
    
    max_beaker_height = 4
    
    tr_x_mtx = np.zeros((len(pairs_tr), max_input_length + max_input_length + (max_beaker_height * 7)))
    tr_y_mtx = np.zeros((len(pairs_tr), max_output_length))
    dev_x_mtx = np.zeros((len(pairs_dev), max_input_length + max_input_length + (max_beaker_height * 7)))
    te_x_mtx = np.zeros((len(pairs_te), max_input_length + max_input_length + (max_beaker_height * 7)))
    
    for i in range(len(pairs_tr)):
        prev_x = pairs_tr[i][0]
        words = prev_x.split(" ")
        for j in range(len(words)):
            tr_x_mtx[i][j] = word_to_idx_input[words[j]] 
        x = pairs_tr[i][1]
        y = pairs_tr[i][2]
        words = x.split(" ")
        for j in range(len(words)):
            tr_x_mtx[i][j + max_input_length] = word_to_idx_input[words[j]] 
        words = y.split(" ")
        for j in range(len(words)):
            tr_y_mtx[i][j] = word_to_idx_output[words[j]]
        initial_env = pairs_tr[i][3]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    tr_x_mtx[i][(max_input_length * 2) + (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue
            
        
    for i in range(len(pairs_dev)):
        prev_x = pairs_dev[i][0]
        words = prev_x.split(" ")
        for j in range(len(words)):
            dev_x_mtx[i][j] = word_to_idx_input[words[j]] 
        x = pairs_dev[i][1]
        y = pairs_dev[i][2]
        words = x.split(" ")
        for j in range(min(len(words), max_input_length)):
            dev_x_mtx[i][j + max_input_length] = word_to_idx_input[words[j]]
        initial_env = pairs_dev[i][3]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    dev_x_mtx[i][(max_input_length * 2)+ (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue
            
    for i in range(len(pairs_te)): 
        prev_x = pairs_te[i][0]
        words = prev_x.split(" ")
        for j in range(len(words)):
            te_x_mtx[i][j] = word_to_idx_input[words[j]] 
        x = pairs_te[i][1]
        y = pairs_te[i][2]
        words = x.split(" ")
        for j in range(min(len(words), max_input_length)):
            te_x_mtx[i][j + max_input_length] = word_to_idx_input[words[j]]  
        initial_env = pairs_te[i][3]
        for j in range(7):
            beaker = initial_env[j]
            offset = 4 - len(beaker)
            for k in range(max_beaker_height):
                if k < len(beaker):
                    te_x_mtx[i][(max_input_length * 2) + (j * max_beaker_height) + (offset + k)] = beaker[k]
                else:
                    continue    
                               
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(tr_x_mtx.astype(int)), torch.from_numpy(tr_y_mtx.astype(int)))
    dev_data = torch.utils.data.TensorDataset(torch.from_numpy(dev_x_mtx.astype(int)), torch.zeros((dev_x_mtx.shape[0], 1)))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(te_x_mtx.astype(int)), torch.zeros((te_x_mtx.shape[0], 1)))
    return train_data, dev_data, test_data, idx_to_word_input, idx_to_word_output


def train(model, train_loader, val_loader, optimizer, epochs):
    """Finds parameters in the model given the training data.

    TODO: implement this function -- suggested implementation iterates over epochs,
        computing loss over training set (in batches, maybe), evaluates on a held-out set
        at each round (you are welcome to split train_data here, or elsewhere), and
        saves the final model parameters.

    Inputs:
        model (Model): The model to train.
        train_data (list of examples): The training examples given.
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0)
    
    no_improvement = 0
    lowest_validation_loss = 100000000
    best_model = copy.deepcopy(model)
    
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        batches = 0
        tic = time.perf_counter()
        for i, batch in enumerate(train_loader):
            loss = model.train_batch(batch, optimizer, criterion)
            epoch_loss += loss.item()
            running_loss += loss.item()
            batches += 1
            
            if i % 120 == 119:    # every 120 mini-batches
                print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / 120))
                running_loss = 0.0
    
        print("Epoch " + str(epoch) + " training loss: " + str(epoch_loss / batches))
        toc = time.perf_counter()
        print("Finished training epoch " + str(epoch) + f" in {toc - tic:0.4f} seconds")
        
        val_loss = 0
        val_running_loss = 0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss = model.train_batch(batch, optimizer, criterion, eval = True)
                val_loss += loss.item()
                val_running_loss += loss.item()
                val_batches += 1
                
                if i % 120 == 119:    # every 120 mini-batches
                    print('[%d, %5d] validation loss: %.3f' % (epoch + 1, i + 1, val_running_loss / 120))
                    val_running_loss = 0.0
        
        print("Epoch " + str(epoch) + " validation loss: " + str(val_loss / val_batches) + "\n\n")
        
        if val_loss / val_batches < lowest_validation_loss:
            lowest_validation_loss = val_loss / val_batches
            best_model = copy.deepcopy(model)
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement == 3:
            break
    
    model = best_model
    model.eval()
            
def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.

    TODO: This code assumes the world state is a string. However, you may sometimes
    start with an AlchemyWorldState object. I suggest loading the AlchemyWorldState objects
    into memory in load_data, and moving that part of the code to load_data. The following
    code just serves as an example of how to 1) make an AlchemyWorldState and 2) execute
    a sequence of actions on it.

    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)

    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]
        
        # JSON file doesn't contain  NO_ARG.
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]

        fsa.feed_complete_action(act, arg1, arg2)

    return fsa.world_state()

def predict(model, data, idx_to_word, outname, split_actions = False):
    """Makes predictions for data given a saved model.

    This function should predict actions (and call the AlchemyFSA to execute them),
    and save the resulting world states in the CSV files (same format as *.csv).

    TODO: you should implement both "gold-previous" and "entire-interaction"
        prediction.

    In the first case ("gold-previous"), for each utterance, you start in the previous gold world state,
    rather than the on you would have predicted (if the utterance is not the first one).
    This is useful for analyzing errors without the problem of error propagation,
    and you should do this first during development on the dev data.

    In the second case ("entire-interaction"), you are continually updating
    a single world state throughout the interaction; i.e. for each instruction, you start
    in whatever previous world state you ended up in with your prediction. This method can be
    impacted by cascading errors -- if you made a previous incorrect prediction, it's hard
    to recover (or your instruction might not make sense). 

    For test labels, you are expected to predict /final/ world states at the end of each
    interaction using "entire-interaction" prediction (you aren't provided the gold
    intermediate world states for the test data).

    Inputs:
        model (Model): A seq2seq model for this task.
        data (list of examples): The data you wish to predict for.
        outname (str): The filename to save the predictions to.
    """
    model = model.to(device)
    model.eval()
    dev_loader = data[0]
    test_loader = data[1]
    dev_out_fn = outname[0]
    test_out_fn = outname[1]
    
    dev_predictions = []
    for i, batch in enumerate(dev_loader):
        if split_actions: 
            prediction = []
            current_action = []
            for idx in model.predict(batch[0]):
                if idx_to_word[idx] == 'SOS':
                    prediction.append(idx_to_word[idx])
                elif idx_to_word[idx] == 'EOS':
                    prediction.append("_".join(current_action))
                    current_action = []
                    prediction.append(idx_to_word[idx])
                elif idx_to_word[idx] == ',':
                    prediction.append("_".join(current_action))
                    current_action = []
                else:
                    current_action.append(idx_to_word[idx])
        else:
            prediction = [idx_to_word[idx] for idx in model.predict(batch[0])]
        dev_predictions.append(prediction) 
        
        if i % 200 == 0:
            print("predicted " + str(i) + " development utterances")
    
    test_predictions = []
    for i, batch in enumerate(test_loader):
        if split_actions:
            prediction = []
            current_action = []
            for idx in model.predict(batch[0]):
                if idx_to_word[idx] == 'SOS':
                    prediction.append(idx_to_word[idx])
                elif idx_to_word[idx] == 'EOS':
                    prediction.append("_".join(current_action))
                    current_action = []
                    prediction.append(idx_to_word[idx])
                elif idx_to_word[idx] == ',':
                    prediction.append("_".join(current_action))
                    current_action = []
                else:
                    current_action.append(idx_to_word[idx])
        else:
            prediction = [idx_to_word[idx] for idx in model.predict(batch[0])]
        test_predictions.append(prediction) 
    
        if i % 200 == 0:
            print("predicted " + str(i) + " test utterances")
        
    id_col,initial_env,after_env,action_col = utils.get_state_id_infor("dev.json")
    dev_instruction_df = utils.instruction_level_prediction(dev_predictions, initial_env,after_env,id_col, drive_path + dev_out_fn + "_instruction.csv")
    dev_interactive_df = utils.interaction_level_prediction(dev_predictions, initial_env,id_col, drive_path + dev_out_fn + "_interaction.csv")
    
    
    id_col,initial_env,after_env,action_col = utils.get_state_id_infor("test_leaderboard.json")
    test_instruction_df = utils.instruction_level_prediction(test_predictions, initial_env,after_env,id_col,drive_path + test_out_fn + "_instruction.csv")
    test_interactive_df = utils.interaction_level_prediction(test_predictions, initial_env,id_col, drive_path + test_out_fn + "_interaction.csv")
    
        
    

def main():
    # A few command line arguments
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--predict", type=bool, default=False)
    parser.add_argument("--saved_model", type=str, default="")
    args = parser.parse_args()

    assert args.train or args.predict

    # Load the data; you can also use this to construct vocabularies, etc.
    train_data = load_data("train.json")
    dev_data = load_data("dev.json")

    # Construct a model object.
    model = Model()

    if args.train:
       # Trains the model
       train(model, train_data) 
    if args.predict:
        # Makes predictions for the data, saving it in the CSV format
        assert args.saved_model

        # TODO: you can modify this to take in a specified split of the data,
        # rather than just the dev data.
        predict(model, dev_data)

        # Once you predict, you can run evaluate.py to get the instruction-level
        # or interaction-level accuracies.

if __name__ == "__main__":
    #main()
    split_actions = True
    batch_size = 20
    
    part4 = True
    
    if part4:
        train_data, dev_data, test_data, idx_to_word_input, idx_to_word_output = load_data_part4("train.json", "dev.json", "test_leaderboard.json")
        # 2700 is roughly 15% of the number of total training set utterances
        train_data, val_data = torch.utils.data.random_split(train_data, [len(train_data) - 2700, 2700], generator=torch.Generator().manual_seed(1))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)
        dev_loader = torch.utils.data.DataLoader(dev_data, shuffle = False, batch_size = 1)
        test_loader = torch.utils.data.DataLoader(test_data, shuffle = False, batch_size = 1)
    else:
        train_data, dev_data, test_data, idx_to_word_input, idx_to_word_output = load_data("train.json", "dev.json", "test_leaderboard.json", split_actions)
        # 2700 is roughly 15% of the number of total training set utterances
        train_data, val_data = torch.utils.data.random_split(train_data, [len(train_data) - 2700, 2700], generator=torch.Generator().manual_seed(1))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)
        dev_loader = torch.utils.data.DataLoader(dev_data, shuffle = False, batch_size = 1)
        test_loader = torch.utils.data.DataLoader(test_data, shuffle = False, batch_size = 1)
    
    
    '''
    part1Model = BasicModel(len(idx_to_word_input) + 1, embedding_size = 64, instruction_context_size = 128, encoder_hidden_size = 128, decoder_hidden_size = 128, output_size = len(idx_to_word_output) + 1).to(device)
    optimizer = torch.optim.Adam(part1Model.parameters(), lr = 0.001)
    train(part1Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part1Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    
    '''
    '''
    part1Model2 = BasicModel2(len(idx_to_word_input) + 1, embedding_size = 64, instruction_context_size = 128, encoder_hidden_size = 128, output_size = len(idx_to_word_output) + 1).to(device)
    optimizer = torch.optim.Adam(part1Model2.parameters(), lr = 0.001)
    train(part1Model2, train_loader, val_loader, optimizer, epochs = 50)
    predict(part1Model2, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    
    '''
 
    '''
    part2Model = BasicModelWithEnvironment(input_size = len(idx_to_word_input) + 1, embedding_size = 50, instruction_encoding_size = 64, state_encoding_size = 16, output_size = len(idx_to_word_output) + 1).to(device)
    optimizer = torch.optim.Adam(part2Model.parameters(), lr = 0.001, weight_decay = 1e-5)
    train(part2Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part2Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    '''
    
    '''
    part2Model = Part2Model(input_size = len(idx_to_word_input) + 1, 
                            word_embedding_size = 64, 
                            action_embedding_size = 64,
                            instruction_context_size = 128, 
                            encoder_hidden_size = 128, 
                            environment_context_size = 16, 
                            environment_hidden_size = 16, 
                            decoder_hidden_size = 128, 
                            output_size = len(idx_to_word_output) + 1).to(device)
    optimizer = torch.optim.Adam(part2Model.parameters(), lr = 0.001)
    train(part2Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part2Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    '''
    '''
    part2Model = Part2Model(input_size = len(idx_to_word_input) + 1, 
                            word_embedding_size = 64, 
                            action_embedding_size = 64,
                            instruction_context_size = 128, 
                            encoder_hidden_size = 128, 
                            environment_context_size = 16, 
                            environment_hidden_size = 16, 
                            decoder_hidden_size = 128, 
                            output_size = len(idx_to_word_output) + 1,
                            one_hot = True).to(device)
    optimizer = torch.optim.Adam(part2Model.parameters(), lr = 0.001)
    train(part2Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part2Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    
    '''
    '''
    part3Model = Part3_BahdanauAttention_Model(input_size = len(idx_to_word_input) + 1, 
                            word_embedding_size = 64, 
                            action_embedding_size = 64, 
                            encoder_hidden_size = 128, 
                            environment_context_size = 16, 
                            environment_hidden_size = 16, 
                            decoder_hidden_size = 128, 
                            output_size = len(idx_to_word_output) + 1,
                            one_hot = True).to(device)
    optimizer = torch.optim.Adam(part3Model.parameters(), lr = 0.001)
    train(part3Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part3Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    '''
    '''
    part3Model = Part3_BasicDotProductAttention_Model(input_size = len(idx_to_word_input) + 1, 
                            word_embedding_size = 64, 
                            action_embedding_size = 64, 
                            encoder_hidden_size = 128, 
                            environment_context_size = 16, 
                            environment_hidden_size = 16, 
                            decoder_hidden_size = 128, 
                            output_size = len(idx_to_word_output) + 1,
                            one_hot = True).to(device)
    optimizer = torch.optim.Adam(part3Model.parameters(), lr = 0.001)
    train(part3Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part3Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    '''
    
    part4Model = Part4_BahdanauAttention_Model(input_size = len(idx_to_word_input) + 1, 
                            word_embedding_size = 64, 
                            action_embedding_size = 64, 
                            encoder_hidden_size = 128, 
                            environment_context_size = 16, 
                            environment_hidden_size = 16, 
                            decoder_hidden_size = 128, 
                            output_size = len(idx_to_word_output) + 1,
                            one_hot = True).to(device)
    optimizer = torch.optim.Adam(part4Model.parameters(), lr = 0.001)
    train(part4Model, train_loader, val_loader, optimizer, epochs = 50)
    predict(part4Model, (dev_loader, test_loader), idx_to_word_output, ("dev_predictions","test_predictions"), split_actions)
    
    