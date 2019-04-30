import math

def process_string(input_str):
    input_str = input_str.replace(' ', '').lower()
    input_len = len(input_str)
    row_lower_limit = math.floor(math.sqrt(input_len))
    higher_column_limit = math.ceil(math.sqrt(input_len))
    return input_str, input_len, row_lower_limit, higher_column_limit

def create_cipher(input_str):
    input_str, input_len, row_lower_limit, higher_column_limit = process_string(input_str)
    words = []
    total_len = 0
    for i in range(0, input_len, higher_column_limit):
        total_len += higher_column_limit
        empty_strings = []
        if total_len - input_len > 0:
            empty_strings = [''] * (total_len - input_len)
        words.append(list(input_str[i:i+higher_column_limit]) + empty_strings)
    words = [''.join([row[i] for row in words]) for i in range(0, higher_column_limit)]
    return ' '.join(words)

input_str = 'On a scale from one to ten what is your favourite colour of the alphabet'
print(create_cipher(input_str))

test_cases = [
    ('On a scale from one to ten what is your favourite colour of the alphabet', 
     'ofoivohe nrtsolet aoeyuoa smnorul cowuirp anhrtoh leafefa ettactb'),
    ('lookadistraction', 'latt odri oiao kscn'),
    ('bananaerror', 'bnr aao ner ar'),
    ('chillout', 'clu hlt io'),
    ('    i      ', 'i'),
    (' i m xy ', 'ix my'),
    ('i m x,y', 'i, my x')
]
for test_case in test_cases:
    try:
        assert create_cipher(test_case[0]) == test_case[1]
    except:
        print('Test case failed for test case ', ' => '.join(test_case))



