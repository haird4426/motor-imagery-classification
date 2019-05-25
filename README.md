# Motor-Imagery-Tasks-Classification-using-EEG-data

In this project, datasets collected from electroencephalography (EEG) are used. A complete description of the data is available at: http://www.bbci.de/competition/iv/desc_2a.pdf

EEG reflects the coordinated activity of millions of neurons near a non-invasive scalp electrode. Because these are scalp potentials, necessarily, they have relatively poor spatiotemporal resolution compared to other neural recording techniques. EEG is believed to be recording dipoles that are transmitted through the scalp.

For each subject, response from 25 EEG electrodes is recorded, while the user imagines performing one of four actions. 22 of these are Electroencephalogram (EEG) while the rest 3 are Electrooculography (EOG). Therefore, this is a classification task (with four outcome classes), where the data is used to determine what action the subject was imagining. The data from only the EEG is used in this project. The performance of CNNs, RNNs, LSTM, GRU and Bidirectional RNNs with different architectures were experimented.
