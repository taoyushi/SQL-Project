# utils/casa_data_utils.py
import torch

def prepare_batch_inputs_and_labels(batch, tokenizer):
    batch_size = len(batch)
    
    batch_questions = [data[0] for data in batch]
    
    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]

    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]
    
    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]
        table_names_in_one_db = batch_table_names[batch_id]
        column_infos_in_one_db = batch_column_infos[batch_id]

        batch_column_number_in_each_table.append([len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []
        
        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")
            
            for column_info in column_infos_in_one_db[table_id]:
                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")
            
            input_tokens = input_tokens[:-1]
        
        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(
        batch_input_tokens, 
        return_tensors="pt", 
        is_split_into_words = True, 
        padding = "max_length",
        max_length = 512,
        truncation = True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []
    
    # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    for batch_id in range(batch_size):
        word_ids = tokenized_inputs.word_ids(batch_index = batch_id)

        aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
        aligned_table_labels, aligned_column_labels = [], []

        # align question tokens
        for token_id, word_id in enumerate(word_ids):
            if word_id == 0:
                aligned_question_ids.append(token_id)

        # align table names
        for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if table_name_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current table name
            if len(temp_list) != 0:
                aligned_table_name_ids.append(temp_list)
                aligned_table_labels.append(batch_table_labels[batch_id][t_id])

        # align column names
        for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if column_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current column name
            if len(temp_list) != 0:
                aligned_column_info_ids.append(temp_list)
                aligned_column_labels.append(batch_column_labels[batch_id][c_id])

        batch_aligned_question_ids.append(aligned_question_ids)
        batch_aligned_table_name_ids.append(aligned_table_name_ids)
        batch_aligned_column_info_ids.append(aligned_column_info_ids)
        batch_aligned_table_labels.append(aligned_table_labels)
        batch_aligned_column_labels.append(aligned_column_labels)

    # update column number in each table (because some tables and columns are discarded)
    for batch_id in range(batch_size):
        if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
            batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][ : len(batch_aligned_table_labels[batch_id])]
        
        if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
            truncated_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(batch_aligned_column_labels[batch_id])
            batch_column_number_in_each_table[batch_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    return encoder_input_ids, encoder_input_attention_mask, \
        batch_aligned_column_labels, batch_aligned_table_labels, \
        batch_aligned_question_ids, batch_aligned_column_info_ids, \
        batch_aligned_table_name_ids, batch_column_number_in_each_table