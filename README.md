# Spotlight

Okay so you know how sometimes people in your lecture get a little too loud and it distracts you from what the prof is saying?
Spotlight is my attempt to fix that. <br>
<br>
Basically I want to: <br>
1. record the lecture through my phone
2. decompose some number of millisecond of the recording into it's frequencies using a fast fourier transform
3. filter the frequencies using a neural net to remove the sound of anyone talking, except for the prof
4. play that through my noise cancelling headphones so that I only hear the prof <br>
<br>
I'm in the process of figuring out the best way to collect training data for this but it's most likely going to be some combination of recording people talking in my physical lectures and artifically creating data to simulate real lectures by throwing sound clips of people talking onto lectures recorded online. <br>
<br>
Current approach to this is to use a basic feed-forward denoising autoencoder and train it by providing noisy audio samples and calcuating the loss with the corresponding clean version. <br>
If that doesn't work I'll try a RNN or something.
