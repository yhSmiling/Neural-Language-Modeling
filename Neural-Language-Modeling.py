import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re

torch.manual_seed(1)

class NGramLanguageModeler(nn.Module):#build neural network

    def __init__(self, vocab_size, embedding_dim, context_size):##vocab_size 是所有的单词数   embedding_dim是
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)#first layer of neural network
        self.linear2 = nn.Linear(128, vocab_size)##second layer of neural network

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def splitSentence(sentence):
	store_word_list=[]
	split_sentence=whole_sentence.split('.')#split sentence according to "."
	count=0
	for i in split_sentence:
		word=i.split(' ')
		if count<len(split_sentence)-1:
			store_word_list.append(word)
			count=count+1

	sentence_list=[]		
	for i in store_word_list:
		while '' in i:
			i.remove('')
		sentence_list.append(i)
	return sentence_list

def get_trigrams(sentence):##get the trigrams
	trigrams_single_sentence = [([sentence[i], sentence[i + 1]], sentence[i + 2])
            	for i in range(len(sentence) - 2)]
	return trigrams_single_sentence

def get_model(trigrams,word_to_ix):
	for epoch in range(100):
	    total_loss = torch.Tensor([0])
	    for context, target in trigrams:
	        # Step 1. Prepare the inputs to be passed to the model
	        # into integer indices and wrap them in variables)
	        context_idxs = [word_to_ix[w] for w in context]
	        # print(context_idxs)
	        context_var = autograd.Variable(torch.LongTensor(context_idxs))
	        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
	        # new instance, you need to zero out the gradients from the old
	        # instance
	        model.zero_grad()
	        # Step 3. Run the forward pass, getting log probabilities over next  
	        # words
	        # print(context_var)
	        log_probs = model(context_var)

	        # Step 4. Compute your loss function. (Again, Torch wants the target
	        # word wrapped in a variable)
	        loss = loss_function(log_probs, autograd.Variable(
	            torch.LongTensor([word_to_ix[target]])))

	        # Step 5. Do the backward pass and update the gradient
	        loss.backward()
	        optimizer.step()

	        total_loss += loss.data
	    losses.append(total_loss)
	return model

def get_vocab(whole_sentence):
	sentence_word=re.split(r'\W', whole_sentence)
	vocab = set(sentence_word)#去重
	vocab.remove('')
	return vocab

def get_predict_model(trigrams_single_sentence_q1):
	list_for_prob=[]
	for context,target in trigrams_single_sentence_q1:
		context_idxs = [word_to_ix[w] for w in context]
		context_var = autograd.Variable(torch.LongTensor(context_idxs))

		log_probs=model(context_var)
		max_prob=torch.max(log_probs,1)[1]
		list_for_prob.append(int(max_prob))

	return list_for_prob
def first_question(word_to_ix):
	test_sentence_question_1=""""Start The mathematician ran to the store ."""
	sentence_word_test_1=re.split(r'\W', test_sentence_question_1)
	while '' in sentence_word_test_1:
		sentence_word_test_1.remove('')
	trigrams_single_sentence_q1 = get_trigrams(sentence_word_test_1)
	
	list_for_prob=get_predict_model(trigrams_single_sentence_q1)

	list_for_predict_word=[]

	count=0
	for i in list_for_prob:
		if i in ix_to_word.keys():
			list_for_predict_word.append(ix_to_word[i])
	print("The first question:")
	print("The predict sentence with 100 Iteration and 0.001 lr value is :",list_for_predict_word)
	# print("The predict sentence with 10 Iteration and 0.001 lr value is :",list_for_predict_word)
	# print("The predict sentence with 100 Iteration and 1 lr value is :",list_for_predict_word)

def second_question(word_to_ix):
	context=['Start','The']
	context_idxs=[word_to_ix[w] for w in context]
	context_var = autograd.Variable(torch.LongTensor(context_idxs))
	log_probs=model(context_var)
	max_prob=torch.max(log_probs,1)[1]
	max_prob=int(max_prob)
	print("The second question:")
	print("The answer more likely to fill in is :",ix_to_word[max_prob])


if __name__ == '__main__':


	CONTEXT_SIZE = 2
	EMBEDDING_DIM = 5
	whole_sentence="""Start The mathematician ran .Start The mathematician ran to the store .Start The physicist ran to the store .Start The philosopher thought about it .Start The mathematician solved the open problem ."""

	sentence_list=splitSentence(whole_sentence)

	trigrams=[]

	for sentence in sentence_list:#use trigrams to store all training word.
		trigrams_single_sentence=get_trigrams(sentence)
		trigrams.extend(trigrams_single_sentence)

	vocab=get_vocab(whole_sentence)

	word_to_ix = {word: i for i, word in enumerate(vocab)}#build the index for every single word
	losses = []
	loss_function = nn.NLLLoss()##loss function
	model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)##get the model
	optimizer = optim.SGD(model.parameters(), lr=0.001)
	model=get_model(trigrams,word_to_ix)
	ix_to_word={v:k for k,v in word_to_ix.items()}
	firstQuestion=first_question(word_to_ix)#get the output for the first question 
	secondQuesion=second_question(word_to_ix)#get the output for the second question
	










