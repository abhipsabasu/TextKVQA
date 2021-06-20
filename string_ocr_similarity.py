####
# first install these
# $ pip install -U strsimpy
# $ pip install unidecode
###

from strsimpy.jaccard import Jaccard  # IOU
import nltk
from nltk.corpus import stopwords
import json
import re
import unidecode

jaccacard = Jaccard(k=3)  # k character long n-gram
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))


def clean_name(name):
    '''
    To remove non alphanumeric characters and accents from title string so that it can be compared with other names
    Also lowering all the letters
    e.g. : 'Toy_Story' --> 'toy story'
    '''
    accent_removed = unidecode.unidecode(name)
    return re.sub(r'[^a-zA-Z0-9 ]+', '', accent_removed).strip().lower()


def remove_english_stopwords(string, english_stopwords=english_stopwords):
    filtered_words = [word for word in string.split() if word not in english_stopwords]
    return filtered_words


def get_title_score(title, ocr_info, similarity_fn=jaccacard.similarity):
    ''' Takes title and ocr_info of an Image. Gives similarity score of title and the closest matching OCR token.
    '''

    clean_title = clean_name(title)

    title_words = remove_english_stopwords(clean_title)

    # taking only the max score for each title word : i.e. out of all ocr words, take only the closest
    title_score = 0
    ocr_word_info_with_max_score = None
    ocr_word_max_score = 0
    for title_word in title_words:
        title_word_score_max = 0
        for ocr_word_info in ocr_info:
            ocr_word = clean_name(ocr_word_info['word'])

            if ocr_word in english_stopwords:
                continue

            confidence = ocr_word_info['confidence']
            ocr_bbox_norm_area = (ocr_word_info['norm_bbox'][2] - ocr_word_info['norm_bbox'][0]) * (
                        ocr_word_info['norm_bbox'][3] - ocr_word_info['norm_bbox'][1])

            title_word_score = confidence * similarity_fn(ocr_word, title_word)

            if title_word_score > title_word_score_max:
                title_word_score_max = title_word_score
                if title_word_score > ocr_word_max_score:
                    ocr_word_max_score = title_word_score
                    ocr_word_info_with_max_score = ocr_word_info

        title_score += title_word_score_max

    avg_title_score = title_score / len(title_words)
    return avg_title_score, ocr_word_info_with_max_score

# # to get title score
#similarity_fn = jaccacard.similarity
#title = "Bharti Airtel"
#with open('ocr_with_bbox.json') as f:
#    text = json.load(f)
#with open("KB-business.json") as f:
#    kb_json = json.load(f)
#ocr_info = text['Q854867_17']
#title_score_dict = {}
#for knowledge_key in kb_json:
#    title = kb_json[knowledge_key]["has title"]
#    title_score, ocr_word_info_with_max_score = get_title_score(title, ocr_info, similarity_fn)
#    if title_score == 0.0:
#        continue
#    title_score_dict[knowledge_key] = title_score
#title_score_dict = {k: v for k, v in sorted(title_score_dict.items(), key=lambda item: item[1], reverse=True)}
#title_score, ocr_word_info_with_max_score = get_title_score(title, ocr_info, similarity_fn)
#print(title_score)
#print(ocr_word_info_with_max_score)
#print(title_score_dict)