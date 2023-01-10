import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import preprocessing

# dev
def plot_audio_time_series(series, title):
    plt.figure(figsize=(18,5))

    time = np.linspace(0, len(series), len(series)) 
    plt.plot(time, series)
    plt.xlabel("Audio Time Series")
    plt.title(title)

    plt.show()


def plot_magnitude_spectrum(signal, title, sr):
    ft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(ft)

    # plot magnitude spectrum
    plt.figure(figsize=(18, 5))

    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    plt.plot(frequency, magnitude_spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.title(title)

    plt.show()


########### Load in audio files ###########
audio_dir_clean = '../assets/data/Clean/ESC195_lecture_32_clean.wav'
audio_dir_noisy = '../assets/data/Noisy/ESC195_lecture_32_noisy.wav'
#audio_dir_clean = '../assets/khan_clean.wav' # just for initial dev ^ takes too long
#audio_dir_noisy = '../assets/khan_noisy.wav'

audio_clean, sr = librosa.load(audio_dir_clean, sr=10000)
audio_noisy, sr = librosa.load(audio_dir_noisy, sr=10000)

# dev
#plot_audio_time_series(audio_clean, "Clean Audio Time Series")
#plot_audio_time_series(audio_noisy, "Noisy Audio Time Series")

#plot_magnitude_spectrum(audio_clean, "Clean Audio FFT", sr)
#plot_magnitude_spectrum(audio_noisy, "Noisy Audio FFT", sr)

# Split into groups of 20 milliseconds worth of samples
division_const = 100
num_samples = sr//division_const # int num of samples in 20 milliseconds
num_divisions = len(audio_clean)//num_samples

audio_clean = audio_clean[:num_samples*num_divisions] # cut off leftovers
audio_noisy = audio_noisy[:num_samples*num_divisions]

clean_samples = np.array(np.split(audio_clean, num_divisions))
noisy_samples = np.array(np.split(audio_noisy, num_divisions))

########### Process the data (FFT, normalize, etc.) ###########
clean_processed = np.zeros(shape=(num_divisions, int((sr/division_const)//2)))
noisy_processed = np.zeros(shape=(num_divisions, int((sr/division_const)//2)))

for i in range(num_divisions):
    # Take data point
    clean_point = clean_samples[i]
    noisy_point = clean_samples[i]

    # FFT
    clean_point = np.abs(np.fft.fft(clean_point))
    noisy_point = np.abs(np.fft.fft(noisy_point))

    clean_point = clean_point[: len(clean_point)//2] # Cut in half to reduce data b/c FFT
    noisy_point = noisy_point[: len(noisy_point)//2] # gives a lower and higher HZ copy of freqs

    """# Normalize
    scaler = preprocessing.StandardScaler()

    scaler.fit(clean_point)
    clean_point = scaler.transform(clean_point)

    scaler.fit(noisy_point)
    noisy_point = scaler.transform(noisy_point)"""

    # Ln the data
    clean_point = np.log(clean_point) # Remember to exp to reconstruct sound
    noisy_point = np.log(noisy_point)

    # Save
    clean_processed[i] = clean_point
    noisy_processed[i] = noisy_point

########### Randomize data ###########
randomize = np.arange(len(clean_processed)) # need to shuffle both arrays in unison
np.random.shuffle(randomize)
clean_processed = clean_processed[randomize]
noisy_procssed = noisy_processed[randomize]

########### Split into training, testing, and evaluation data ###########
# Define percentages
TRAINING = 0.60
VALIDATION = 0.15
TESTING = 0.25

total_len = clean_processed.shape[0]
training_idx = int(total_len*TRAINING)
validation_idx = training_idx + int(total_len*VALIDATION)

training_set_clean = clean_processed[0:training_idx]
validation_set_clean = clean_processed[training_idx:validation_idx]
testing_set_clean = clean_processed[validation_idx:]

training_set_noisy = noisy_processed[0:training_idx]
validation_set_noisy = noisy_processed[training_idx:validation_idx]
testing_set_noisy = noisy_processed[validation_idx:]

print(f"There are: \
        \n\t{len(training_set_clean)} training samples \
        \n\t{len(validation_set_clean)} validation samples \
        \n\t{len(testing_set_clean)} testing samples")

########### Save data into txt files ###########
np.savetxt('../assets/data/clean/training_clean.txt', training_set_clean)
np.savetxt('../assets/data/clean/validation_clean.txt', validation_set_clean)
np.savetxt('../assets/data/clean/testing_clean.txt', testing_set_clean)

np.savetxt('../assets/data/noisy/training_noisy.txt', training_set_noisy)
np.savetxt('../assets/data/noisy/validation_noisy.txt', validation_set_clean)
np.savetxt('../assets/data/noisy/testing_noisy.txt', testing_set_noisy)
