
import json
import csv 

def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    
    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0,1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)
    
    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words

def compute_grad_norm(parameters, norm_type=2):
    """ Ref: http://pytorch.org/docs/0.3.0/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_state_id_infor(filename):
    print("Extracting infor from" + str(filename))
    with open(filename) as f:
        data = json.load(f)
    after_env = []
    id_col = []
    initial_env = []
    action_col = []
    for i in range(len(data)):
        cur_id = data[i]["identifier"]
        cur_init_env = data[i]["initial_env"]
        cur_lst  = data[i]["utterances"]
      
        for j in range(len(cur_lst)):
            cur_dict = cur_lst[j]
            if "actions" in cur_dict.keys():
                cur_y = cur_dict["actions"]
                adj_cur_y_col = []
                for l in cur_y:
                    #adj_cur_y_col.append(" ".join(l.split()))
                    adj_cur_y_col.append("_".join(l.split())) 
                adj_cur_y = " ".join(adj_cur_y_col)
            
                cur_y = adj_cur_y + " EOS"
                action_col.append(cur_y)

            if "after_env" in cur_dict.keys():
                cur_after_env = cur_dict["after_env"]    

                after_env.append(cur_after_env)
            id_col.append(cur_id)
            initial_env.append(cur_init_env)
         
    return id_col,initial_env,after_env,action_col

from alchemy_world_state import AlchemyWorldState
from alchemy_fsa import AlchemyFSA,token_is_beaker,FSAStates

from fsa import ExecutionFSA, EOS, ACTION_SEP, NO_ARG
ACTION_POP = "pop"
def action_to_state(world_state,action_sequence): #input a single sequence of actions and outputs the single state, and a None indicator

    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)
    ret = None
    for action in action_sequence[:-1]:
        split = action.split("_")
         
        if len(split) < 2: # invalid action
            print("\nInvalid action: ", str(action))
            print("From sequence: ", str(action_sequence[:-1]), "\n")
            return world_state, True #if the converted state is none, then use the previous step true state
        
        act = split[0]
        arg1 = split[1]
        
        # JSON file doesn't contain  NO_ARG.
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]
        ret = fsa.peek_complete_action(act, arg1, arg2)
    if ret == None:
        return world_state, True #if the converted state is none, then use the previous step true state
    else: 
        return ret, False
  
def action_to_state_iter(id_col, initial_env,predicted_train):
    #input a collection of sequence of actions and outputs the collection state
    count_none = 0
    state_col = []
    prev_id = ""
    i = 0
    while i < len(id_col):
        #only need to execuate the last action of each example
        cur_id =id_col[i]
        if cur_id != prev_id:
            prev_id = cur_id
            a, isNone = action_to_state(initial_env[i], predicted_train[i]) 
            if isNone:
                count_none += 1
      
        else:
            a, isNone = action_to_state(after_env[i-1], predicted_train[i]) 
            if isNone:
                count_none += 1
        state_col.append(a)
        i += 1
    print("%s of the actions converted to state encountered none result"%(count_none/len(id_col)))
    return state_col
  
import pandas as pd
def instruction_level_prediction(action_prediction, initial_env,after_env,id_col, out_name): #use the example id, predicted action, and inial state, end state to generate state predictions on instruction level
    count_none = 0
    output_dict = {"id":[], "final_world_state":[]}
    prev_id = ""
    i = 0
    cur_count = 0
    while i < len(id_col):
        
        #only need to execuate the last action of each example
        cur_id =id_col[i]
        if cur_id != prev_id:
            cur_count = 0
            
            prev_id = cur_id
            a, isNone = action_to_state(initial_env[i], action_prediction[i])
            if isNone:
                count_none += 1
      
        else:
            cur_count += 1
            a, isNone = action_to_state(after_env[i-1], action_prediction[i]) 
            if isNone:
                count_none += 1
        output_dict["final_world_state"].append(str(a))
        output_dict["id"].append(cur_id + "-"+str(cur_count))
        i += 1
    
    out_file = pd.DataFrame(output_dict)
    out_file.to_csv(out_name, index = False)
    print("successfully writeout")
    print("%s of the actions converted to state encountered none result"%(count_none/len(id_col)))
    return out_file

def interaction_level_prediction(action_prediction, initial_env,id_col, out_name):
    #use the example id, predicted action, and initial state to generate state predictions on instruction level
    count_none = 0
    count_none = 0
    ongoing_predicted_states = []
    output_dict = {"id":[], "final_world_state":[]}
    prev_id = id_col[0]
    
    id_col.append(None)
    i = 1
#     print(action_to_state(initial_env[0], action_prediction[0]))
    cur_state, isNone = action_to_state(initial_env[0], action_prediction[0])
    if isNone:
        count_none += 1
    
    ongoing_predicted_states.append(str(cur_state) )
    while i < len(id_col):

        cur_id =id_col[i]
        if cur_id != prev_id:
            
            output_dict["final_world_state"].append(str(ongoing_predicted_states[i-1]))
            output_dict["id"].append(prev_id)
            if cur_id != None:
                cur_state, isNone = action_to_state(initial_env[i], action_prediction[i]) 
                if isNone:
                    count_none += 1
                ongoing_predicted_states.append(str(cur_state))
            
            
            prev_id = cur_id
#          

      
        else:
        
            cur_state, isNone = action_to_state(str(ongoing_predicted_states[i-1]), action_prediction[i]) 
            if isNone:
                count_none += 1
            ongoing_predicted_states.append(str(cur_state))
        i += 1
    
    out_file = pd.DataFrame(output_dict)
    out_file.to_csv(out_name, index = False)
    print("successfully writeout")
    print("%s of the actions converted to state encountered none result"%(count_none/len(id_col)))
    return out_file 
  
