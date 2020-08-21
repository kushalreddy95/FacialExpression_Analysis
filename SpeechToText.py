import os
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

os.chdir("../..")

os.chdir("Voice_Memos/")

print(os.getcwd())

users = subprocess.check_output('ls').splitlines()
r = sr.Recognizer()



for user in users:
    print(user.decode("utf-8"))
    os.chdir("{}/".format(user.decode("utf-8")))
    folders = subprocess.check_output('ls').splitlines()
    for folder in folders:
        print(folder.decode("utf-8"))
        if folder.decode("utf-8").startswith("AR"):
            os.chdir("{}/".format(folder.decode("utf-8")))
            audio_files = subprocess.check_output('ls').splitlines()
            for file in audio_files:
                print(file.decode("utf-8"))
                text = ""
                try:
                    if str(file.decode("utf-8")).startswith("audio"):
                        audiocheck = True

                    elif str(file.decode("utf-8").split('.')[1]) == "wav":
                        song = AudioSegment.from_wav(file.decode("utf-8"))
                        # print(song)
                        fh = open("VoiceToText.txt", "a+")

                        n = song.dBFS
                        n = (n % 5)
                        print(n)



                        chunks = split_on_silence(song, min_silence_len=250, silence_thresh=song.dBFS-n)
                        print(song.dBFS)
                        print(len(chunks))
                        try:
                            os.mkdir('audio_chunks')
                        except(FileExistsError):
                            pass

                        os.chdir('audio_chunks')
                        i = 0

                        for chunk in chunks:
                            print("Chunk " + str(i))
                            chunk_silent = AudioSegment.silent(duration=10)
                            audio_chunk = chunk_silent + chunk + chunk_silent
                            print("saving chunk{0}.wav".format(i))
                            audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
                            filename = 'chunk' + str(i) + '.wav'
                            print("Processing chunk " + str(i))
                            obj = filename
                            with sr.AudioFile(obj) as source:
                                # remove this if it is not working
                                # correctly.
                                r.adjust_for_ambient_noise(source)
                                audio_listened = r.listen(source)

                            try:
                                # try converting it to text
                                rec = r.recognize_google(audio_listened)
                                # write the output to the file.
                                fh.write(rec + ". ")

                                # catch any errors.
                            except sr.UnknownValueError:
                                print("Could not understand audio")
                            i+=1

                        os.chdir("..")
                        fh.close()

                except Exception as e:
                    print("Error {}".format(e))

            os.chdir("..")

        print("\n")

    print("\n")
    os.chdir("..")



for user in users:
    if(user.decode("utf-8") == "User2"):
        os.chdir("{}/".format(user.decode("utf-8")))
        folders = subprocess.check_output('ls').splitlines()
        for folder in folders:
            if folder.decode("utf-8")=="AR":
                os.chdir("{}/".format(folder.decode("utf-8")))
                audio_files = subprocess.check_output('ls').splitlines()
                for file in audio_files:
                    try:
                        if str(file.decode("utf-8")).startswith("audio"):
                            audiocheck = True
                        elif str(file.decode("utf-8").split('.')[1]) == "wav":
                            song = AudioSegment.from_wav(file.decode("utf-8"))

                            fh = open("IntSpeechToText.txt", "a+")

                            n = song.dBFS
                            n = (n % 5)
                            print(n)
                            print(song.dBFS)
                            chunks = split_on_silence(song, min_silence_len=250, silence_thresh=song.dBFS - n)

                            print(len(chunks))
                            try:
                                os.mkdir('audio_chunks')
                            except(FileExistsError):
                                pass

                            os.chdir('audio_chunks')
                            i = 0

                            for chunk in chunks:
                                print("Chunk " + str(i))
                                chunk_silent = AudioSegment.silent(duration=10)
                                audio_chunk = chunk_silent + chunk + chunk_silent
                                print("saving chunk{0}.wav".format(i))
                                audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
                                filename = 'chunk' + str(i) + '.wav'
                                print("Processing chunk " + str(i))
                                obj = filename
                                with sr.AudioFile(obj) as source:
                                    # remove this if it is not working
                                    # correctly.
                                    # r.adjust_for_ambient_noise(source)
                                    audio_listened = r.listen(source)

                                try:
                                    # try converting it to text
                                    rec = r.recognize_google(audio_listened)
                                    # write the output to the file.
                                    fh.write(rec + ". ")

                                    # catch any errors.
                                except sr.UnknownValueError:
                                    print("Could not understand audio")
                                i+=1

                            os.chdir("..")
                            fh.close()

                    except Exception as e:
                        print("Error {}".format(e))
                    except Exception as e:
                        print(e)

# para = text.split('.')
# total_polarity = 0
# for item in para:
#     print(item)
#
#     blob1 = TextBlob(item)
#     total_polarity += blob1.sentiment.polarity
#     print(total_polarity)
#     print("\n")
# print(total_polarity)
