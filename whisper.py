from transformers import AutoFeatureExtractor,WhisperModel
import librosa
import numpy as np
import torch

class whisper_Wrapper():
    def __init__(self, device = 'cuda', sampling_rate = 16000, 
                 model_path = './pretrained_weights/wav2vec2-base-960h'
        ) -> None:
        self.device = device

        audio_encoder = WhisperModel.from_pretrained(model_path, local_files_only=True).to(device=device, dtype=torch.float32)
        #audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder = audio_encoder.eval()
        
        self._processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)

        self._sampling_rate = sampling_rate
        
    def forward(self, wav_file, only_last_features = False):
        speech_array, sampling_rate = librosa.load(wav_file, sr=self._sampling_rate)
        
        audio_feature = self._processor(
            speech_array, 
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(self.device)#1*80*3000, 对应30s，所以25fps对应的步长是3000/30/25=4
        num_frames=len(speech_array) // 640 
        audio_feats = self.audio_encoder.encoder(audio_feature[:, :, :3000], output_hidden_states=True).hidden_states
        audio_feats = torch.stack(audio_feats, dim=2) #1*1500*5*384，步长变为2
        audio_feats = torch.cat([torch.zeros_like(audio_feats[:,:4]), audio_feats], 1)
        
        audio_prompts = []

        for bb in range(1):
            audio_feats_list = []
            for f in range(num_frames):
                cur_t = f * 2
                audio_clip = audio_feats[bb:bb+1, cur_t: cur_t+10]
                audio_feats_list.append(audio_clip)
            audio_feats_list = torch.stack(audio_feats_list, 1)
            audio_prompts.append(audio_feats_list)
        audio_prompts = torch.cat(audio_prompts) #1*99*10*5*384
        return audio_prompts
        # with torch.no_grad():
        #     embeddings = self.audio_encoder(input_value, output_hidden_states=True)
        
        # if only_last_features:
        #     fea = embeddings.last_hidden_state[0].unsqueeze(1)
        # else:
        #     fea = embeddings.hidden_states
        #     fea = torch.stack(fea, dim=2)[0] # 36, 13, 768
        
        # return fea

