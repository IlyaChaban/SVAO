import pytest

MESSAGE = "do not eat all my oranges"
TABLE = 'abcdefghijklmnopqrstuvwxyz '
KEY = 1
 
def caesar_encrypt(message, table, key):
    EncryptedMessage=''
    for letter in message:
        LetterPositionInTable=table.find(letter)
        if key%len(table)+LetterPositionInTable>=len(table):
            EncryptedMessage+=table[LetterPositionInTable+key%len(table)-len(table)]
        else:
            EncryptedMessage+=table[LetterPositionInTable+key%len(table)]
    return EncryptedMessage
 
def caesar_decrypt(message, table, key):
    DecryptedMessage=''
    for letter in message:
        LetterPositionInTable=table.find(letter)
        if key%len(table)+LetterPositionInTable<0:
            DecryptedMessage+=table[LetterPositionInTable-key%len(table)+len(table)]
        else:
            DecryptedMessage+=table[LetterPositionInTable-key%len(table)]
    return DecryptedMessage
    
#test everything
pytest.main()

encrypted_message = caesar_encrypt(MESSAGE,TABLE,KEY)
print(encrypted_message)
decrypted_message = caesar_decrypt(encrypted_message,TABLE,KEY)
print(decrypted_message)