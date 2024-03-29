from os import path
from pocketsphinx.pocketsphinx import *
import pronouncing

if __name__ == '__main__':

    MODELDIR = "./model"
    DATADIR = "./test/data"

    # Create a decoder with certain model
    config = Decoder.default_config()
    config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
    config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
    config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
    config.set_string('-logfn','nul')
    decoder = Decoder(config)

    # Decode streaming data.
    decoder = Decoder(config)
    decoder.start_utt()

    # File names: 'f2bjrop1.0.wav', 'f2bjrop1.1.wav', 'f2btrop6.0.wav'
    stream = open(path.join(DATADIR, 'f2btrop6.0.wav'), 'rb')

    while True:
        buf = stream.read(1024)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            break
    decoder.end_utt()


    # print('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
    siln = ['</s>', '<s>', '<sil>']
    words = []
    for word in [seg.word for seg in decoder.seg()]:
        if word not in siln:
            new_word = ''
            for char in word:
                if char.isalpha() or char == "'":
                    new_word += char
            print('{}: {}'.format(new_word, pronouncing.phones_for_word(new_word)))


