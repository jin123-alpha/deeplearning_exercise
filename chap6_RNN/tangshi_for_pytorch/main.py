import numpy as np
import collections
import torch
import torch.optim as optim

import rnn as rnn_lstm


start_token = 'G'
end_token = 'E'
batch_size = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

#rnn_lstm = RNN_model(batch_sz = batch_size,vocab_len = 100 ,word_embedding = None ,embedding_dim= 100, lstm_hidden_dim=128)

def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    rnn_model = rnn_model.to(DEVICE)

    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    loss_fun = torch.nn.NLLLoss().to(DEVICE)
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = torch.from_numpy(np.expand_dims(x,axis=1)).to(DEVICE)
                y = torch.from_numpy(y).to(DEVICE)
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.detach().cpu().tolist()) # the following  three line can print the output and the prediction
                    print('b_y       ', y.detach().cpu().tolist())   # And you need to take a screenshot and then past is to your homework paper.
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.detach().cpu().tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 ==0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)

    text = ''.join(shige).strip()
    if not text:
        print('生成内容过短，未形成完整诗句。')
        return

    first_line = text.split('。')[0].strip()
    printed_first = ''
    if first_line:
        printed_first = first_line if first_line.endswith('。') else first_line + '。'
        print(printed_first)

    has_output = False
    poem_sentences = text.split('。')
    for s in poem_sentences:
        s = s.strip()
        if s != '' and len(s) > 10:
            line = s + '。'
            if line == printed_first:
                continue
            print(line)
            has_output = True

    # 如果按句号切分后没有任何输出，则强制输出一条兜底文本
    if not has_output:
        if not printed_first:
            print(text if text.endswith('。') else text + '。')


def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)
    rnn_model = rnn_model.to(DEVICE)

    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location=DEVICE))
    rnn_model.eval()

    # 指定开始的字

    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = torch.from_numpy(input).to(DEVICE)
        output = rnn_model(input, is_test=True)
        word = to_word(output.detach().cpu().numpy()[-1], vocabularies)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem



#run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("月"))
pretty_print_poem(gen_poem("海"))
pretty_print_poem(gen_poem("君"))


