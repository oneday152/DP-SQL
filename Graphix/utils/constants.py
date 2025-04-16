PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'

# GRAMMAR_FILEPATH = 'asdl/sql/grammar/sql_asdl_v2.txt'
# SCHEMA_TYPES = ['table', 'others', 'text', 'time', 'number', 'boolean']
# MAX_RELATIVE_DIST = 2
# relations: type_1-type_2-rel_name, r represents reverse edge, b represents bidirectional edge
RELATIONS = ['table-table-generic','table-table-fkb','table-table-fk','table-table-fkr','table-table-identity','column-column-generic','column-column-sametable','column-column-identity',
                 'column-column-fk','column-column-fkr','column-table-has','table-column-has','column-table-pk', 'table-column-pk','question-question-identity', 'question-question-generic', 
                 'question-question-dist1','question-table-nomatch','table-question-nomatch','question-table-exactmatch','table-question-exactmatch','question-table-partialmatch','table-question-partialmatch',
                 'question-column-nomatch','column-question-nomatch','question-column-exactmatch','column-question-exactmatch','question-column-partialmatch','column-question-partialmatch',
                 'question-column-valuematch','column-question-valuematch','question-question-modifier','question-question-argument','table-column-generic','column-table-generic']
