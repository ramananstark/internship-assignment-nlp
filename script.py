# ## Import all the dependent libraries
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import random
import PyPDF2

from keybert import KeyBERT
from transformers import T5ForConditionalGeneration, T5Tokenizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')


def get_mca_questions(context: str):
    if not isinstance(context, str):
        raise TypeError("Input must be of string data type")

    text_data = context
    text_data = nltk.sent_tokenize(context)

    def remove_non_ascii(text):
        return text.encode('ascii', 'ignore').decode()

    def remove_patterns(text):
        pattern = r'chap \d+-\d+\.indd\s+\d+\s+\d+/\d+/\d+\s+\d+:\d+:\d+\s+[APM]+Rationalised\s+\d+-\d+'
        return re.sub(pattern, '', text)

    def parenthesis(text):
        return re.sub(r'\([^)]*\)', '', text)

    def to_lowercase(text):
        return text.lower()

    def replace_numbers(text):
        return re.sub(r'\d+', '', text)

    def remove_space(text):
        return text.strip()

    def remove_punctuations(text):
        punctutations = "''!()[]{};:,.'\<>/?@$%^&_-~`=+#"
        for punc in punctutations:
            text = text.replace(punc, "")
        return text

    def tokened(text):
        return word_tokenize(text)

    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]

    text_list = text_data

    processed_text_data = []

    for text in text_list:
        text = remove_non_ascii(text)
        text = remove_patterns(text)
        text = parenthesis(text)
        text = to_lowercase(text)
        text = replace_numbers(text)
        text = remove_space(text)
        text = remove_punctuations(text)
        tokens = tokened(text)
        tokens = remove_stopwords(tokens)
        processed_text = ' '.join(tokens)
        processed_text_data.append(processed_text)

    # print(processed_text_data)
    concatenated_text_list = [' '.join(processed_text_data)]
    print(concatenated_text_list)

    # def TFIdf(text):
    #     extractor=pke.unsupervised.TFIdf()

    #     extractor.load_document(input=text,language='en')
    #     extractor.candidate_selection()
    #     extractor.candidate_weighting()

    #     keyphrases=extractor.get_n_best(n=10)
    #     return [a_tuple[0] for a_tuple in keyphrases]

    # TFIdf_keywords=TFIdf(processed_text)
    # print(TFIdf_keywords)

    kw_model = KeyBERT()
    KeyBERT_ans = []
    for text in concatenated_text_list:
        keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), top_n=10)
        keyword_texts = [keyword[0] for keyword in keywords]
    # print(keywords)
        KeyBERT_ans.extend(keyword_texts)
    print(KeyBERT_ans)
    # for i in range(len(keywords)):

    gen_model = T5ForConditionalGeneration.from_pretrained(
        'ramsrigouthamg/t5_squad_v1')
    t5_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    def get_question(sentence, answer, model, tknizer, max_question_length=10):
        text = "context: {} answer: {}".format(sentence, answer)
        max_len = 512
        max_new_tokens = 300
        encoder = tknizer.encode_plus(
            text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")

        input_ids, attention_mask = encoder['input_ids'], encoder['attention_mask']
        question_variations = []
        for i in range(5):
            outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=max_len,
                                  max_new_tokens=max_new_tokens
                                  )

            dec = [tknizer.decode(ids, skip_special_tokens=True)
                   for ids in outs]

        Question = dec[0].replace("question:", "")
        Question = Question.strip()

        return Question
    # context="Virat Kohli has captained the Royal Challengers Bangalore (RCB) franchise in the Indian Premier League"
    # answer="Virat Kohli"
    # print(get_question(context,answer,gen_model,t5_tokenizer))

    ques = []
    for i in range(len(KeyBERT_ans)):
        questions = get_question(
            processed_text, KeyBERT_ans[i], gen_model, t5_tokenizer)
        ques.append(questions)

    # from nltk.corpus import wordnet

    questions = ques
    keybert_answers = KeyBERT_ans

    def get_hyponyms(word):
        hyponyms = set()
        synsets = wordnet.synsets(word)

        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponyms.add(lemma.name())

        return list(hyponyms)

    def generate_options(question, keybert_answer):
        words = nltk.word_tokenize(question)
        nouns = [word for (word, pos) in nltk.pos_tag(words)
                 if pos.startswith('N')]

        options = []
        for noun in nouns:
            hyponyms = get_hyponyms(noun)
            options.extend(hyponyms)

        random.shuffle(options)

        options.append(keybert_answer)

        random.shuffle(options)

        options = options[:3]
        options.append(keybert_answer)

        random.shuffle(options)
        while len(options) < 4:
            options.extend(options)

        # options = [str(option) for sublist in options for option in sublist]
        return options[:4]

    mca_questions = []
    for question, keybert_answer in zip(questions, keybert_answers):
        options = generate_options(question, keybert_answer)
        if len(options) < 4:
            additional_options = 4 - len(set(options))

            stock_words = ["all of the above",
                           "none of the above", "fillup......", "NA"]

            options.extend(stock_words[:additional_options])

            options = options[:4]
        formatted_question = f"{question} ->"
        formatted_options = " -> ".join(options)

        formatted_mca_question = f"{formatted_question}{formatted_options}"
        # question_option = [
        #      question,
        #      options
        # ]
        mca_questions.append(formatted_mca_question)

    # for item in mca_questions:
    #     print(item["question"])
    #     for i in range(len(item["options"])):
    #         print(f"({chr(i + 65)}) {item['options'][i]}")

    return mca_questions


def extract_text_from_pdf(pdf_path):
    if not pdf_path.endswith('.pdf'):
        raise ValueError("Please provide a PDF file.")

    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print(
            f"An error occurred while extracting text from the PDF: {str(e)}")
        return None


pdf_file_path = input("Enter the PDF file path: ")
context = extract_text_from_pdf(pdf_file_path)
mca_questions_list = get_mca_questions(context)

# for i in mca_questions_list:
#     print(i)
print(mca_questions_list)
