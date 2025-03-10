from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from temgo import BFBeatTracker
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from librosa.beat import beat_track

class Postprocessor:
    """Postprocessor for the beat and downbeat predictions of the model.
    The postprocessor takes the (framewise) model predictions (beat and downbeats) and the padding mask,
    and returns the postprocessed beat and downbeat as list of times in seconds.
    The beats and downbeats can be 1D arrays (for only 1 piece) or 2D arrays, if a batch of pieces is considered.
    The output dimensionality is the same as the input dimensionality.
    Two types of postprocessing are implemented:
        - minimal: a simple postprocessing that takes the maximum of the framewise predictions,
        and removes adjacent peaks.
        - dbn: a postprocessing based on the Dynamic Bayesian Network proposed by Böck et al.
    Args:
        type (str): the type of postprocessing to apply. Either "minimal" or "dbn". Default is "minimal".
        fps (int): the frames per second of the model framewise predictions. Default is 50.
    """

    def __init__(self, type: str = "minimal", fps: int = 50):
        assert type in ["minimal", "dbn", 'bf', "dp", "sppk", 'plpdp', 'pf'], "Invalid postprocessing type"
        self.type = type
        self.fps = fps
        self.dp = beat_track
            

    def __call__(
        self,
        beat: torch.Tensor,
        downbeat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply postprocessing to the input beat and downbeat tensors. Works with batched and unbatched inputs.
        The output is a list of times in seconds, or a list of lists of times in seconds, if the input is batched.

        Args:
            beat (torch.Tensor): The input beat tensor.
            downbeat (torch.Tensor): The input downbeat tensor.
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The postprocessed beat tensor.
            torch.Tensor: The postprocessed downbeat tensor.
        """
        batched = False if beat.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(beat, dtype=torch.bool)

        # if beat and downbeat are 1D tensors, add a batch dimension
        if not batched:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        # apply sigmoid if the model output is not already a probability
        if self.type == 'minimal':
            beat, downbeat = beat.logit(), downbeat.logit()

        if self.type == "minimal":
            postp_beat, postp_downbeat = self.postp_minimal(
                beat, downbeat, padding_mask
            )
        elif self.type == "dbn":
            postp_beat, postp_downbeat = self.postp_dbn(beat, downbeat, padding_mask)
            
        elif self.type == 'bf':
            postp_beat, postp_downbeat = self.postp_bf(beat, downbeat, padding_mask)

        elif self.type == "dp":
            postp_beat, postp_downbeat = self.postp_dp(beat, downbeat, padding_mask)
        
        elif self.type == 'sppk':
            postp_beat, postp_downbeat = self.postp_sppk(beat, downbeat, padding_mask)

        elif self.type == 'pf':
            postp_beat, postp_downbeat = self.postp_pf(beat, downbeat, padding_mask)
        else:
            raise ValueError("Invalid postprocessing type")

        # remove the batch dimension if it was added
        if not batched:
            postp_beat = postp_beat[0]
            postp_downbeat = postp_downbeat[0]

        # update the model prediction dict
        return postp_beat, postp_downbeat

    def postp_pf(self, beat, downbeat, padding_mask):
        beat_prob = beat
        downbeat_prob = downbeat
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_pf_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_pf_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]
        # build an artificial multiclass prediction, as suggested by Böck et al.
        # again we limit the lower bound to avoid problems with the DBN
        epsilon = 1e-5
        combined_act = np.vstack(
            (
                np.maximum(
                    beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2
                ),
                downbeat_prob.cpu().numpy(),
            )
        ).T
        # run the PF
        pf = particle_filter_cascade(
            beats_per_bar=[3, 4],
            particle_size=1500,
            down_particle_size=250,
            min_bpm=55.0,
            max_bpm=210.0,
            fps=self.fps,
            plot=[],
            mode='offline',
        )
        pf_out = pf.process(combined_act)
        postp_beat = pf_out[pf_out[:, 1] == 2][:, 0]
        postp_downbeat = pf_out[pf_out[:, 1] == 1][:, 0]
        return postp_beat, postp_downbeat

    def postp_bf(self, beat, downbeat, padding_mask):
        beat_prob = beat
        downbeat_prob = downbeat
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_bf_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat


    def _postp_bf_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]

        # again we limit the lower bound to avoid problems with the BF
        epsilon = 1e-5
        beat_prob = np.maximum(beat_prob.cpu().numpy(), epsilon / 2)
        downbeat_prob = downbeat_prob.cpu().numpy()

        # run the BF
        bf = BFBeatTracker(
                mode='offline',
                debug=0,
                minbpm=55, 
                maxbpm=215, 
                fps=self.fps,
                align=False,
                correct=False,
                winsize=1.298,
                P1=0.0177,
                P2=0.288,
                multiscale=True,
            )
        # GTZAN {'winsize': 1.2984452989566901, 'P1': 0.017747262677187684, 'P2': 0.28857410238557735, 'align': False, 'maxbpm': 226.75, 'minbpm': 55.15, 'correct': False, 'offset': 0.011912936688929709})
        postp_beat = np.array(bf(onset_envelope=beat_prob)) + 0.011912936688929709
        postp_downbeat = np.array([]) 
        return postp_beat, postp_downbeat

    def postp_dp(self, beat, downbeat, padding_mask):
        beat_prob = beat
        downbeat_prob = downbeat
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_dp_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat
    
    def _postp_dp_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        beat_prob = padded_beat_prob[mask].cpu().numpy()
        downbeat_prob = padded_downbeat_prob[mask].cpu().numpy()
        beat_prob = beat_prob + downbeat_prob
        _, postp_beat = self.dp(onset_envelope=beat_prob, sr=self.fps, hop_length=1, units='time')
        _, postp_downbeat = _, np.array([])
        return postp_beat, postp_downbeat

    
    def postp_minimal(self, beat, downbeat, padding_mask):
        # concatenate beat and downbeat in the same tensor of shape (B, T, 2)
        packed_pred = rearrange(
            [beat, downbeat], "c b t -> b t c", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = packed_pred.masked_fill(~padding_mask.unsqueeze(-1), -1000)
        # reshape to (2*B, T) to apply max pooling
        pred_logits = rearrange(pred_logits, "b t c -> (c b) t")
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000
        )
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = pred_peaks > 0
        #  rearrange back to two tensors of shape (B, T)
        beat_peaks, downbeat_peaks = rearrange(
            pred_peaks, "(c b) t -> c b t", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_minimal_item, beat_peaks, downbeat_peaks, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_minimal_item(self, padded_beat_peaks, padded_downbeat_peaks, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_peaks = padded_beat_peaks[mask]
        downbeat_peaks = padded_downbeat_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        beat_frame = torch.nonzero(beat_peaks).cpu().numpy()[:, 0]
        downbeat_frame = torch.nonzero(downbeat_peaks).cpu().numpy()[:, 0]
        # remove adjacent peaks
        beat_frame = deduplicate_peaks(beat_frame, width=1)
        downbeat_frame = deduplicate_peaks(downbeat_frame, width=1)
        # convert from frame to seconds
        beat_time = beat_frame / self.fps
        downbeat_time = downbeat_frame / self.fps
        # move the downbeat to the nearest beat
        if (
            len(beat_time) > 0
        ):  # skip if there are no beats, like in the first training steps
            for i, d_time in enumerate(downbeat_time):
                beat_idx = np.argmin(np.abs(beat_time - d_time))
                downbeat_time[i] = beat_time[beat_idx]
        # remove duplicate downbeat times (if some db were moved to the same position)
        downbeat_time = np.unique(downbeat_time)
        return beat_time, downbeat_time

    def postp_dbn(self, beat, downbeat, padding_mask):
        beat_prob = beat
        downbeat_prob = downbeat
        # limit lower and upper bound, since 0 and 1 create problems in the DBN
        epsilon = 1e-5
        beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
        downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_dbn_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_dbn_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]
        # build an artificial multiclass prediction, as suggested by Böck et al.
        # again we limit the lower bound to avoid problems with the DBN
        epsilon = 1e-5
        combined_act = np.vstack(
            (
                np.maximum(
                    beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2
                ),
                downbeat_prob.cpu().numpy(),
            )
        ).T
        # run the DBN
        dbn = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4],
                min_bpm=55.0,
                max_bpm=210.0,
                fps=self.fps,
                transition_lambda=100,
            )
        dbn_out = dbn(combined_act)
        postp_beat = dbn_out[:, 0]
        postp_downbeat = dbn_out[dbn_out[:, 1] == 1][:, 0]
        return postp_beat, postp_downbeat

    def postp_sppk(self, beat, downbeat, padding_mask):
        beat_prob = beat
        downbeat_prob = downbeat
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_sppk_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_sppk_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # https://github.com/SunnyCYC/plpdp4beat 
        beat_prob = padded_beat_prob[mask].cpu().numpy()
        downbeat_prob = padded_downbeat_prob[mask].cpu().numpy()
        beat_prob = beat_prob + downbeat_prob
        # run the SPPK
        from scipy.signal import find_peaks
        beats_spppk_tmp, _ = find_peaks(beat_prob, height = 0.1, 
                                        distance = 7, 
                                        prominence = 0.1)
        downbeat_spppk_tmp, _ = find_peaks(downbeat_prob, height = 0.1, 
                                        distance = 7, 
                                        prominence = 0.1)
        postp_beat = beats_spppk_tmp / self.fps
        postp_downbeat = downbeat_spppk_tmp / self.fps
        return postp_beat, postp_downbeat
        

def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """
    Replaces groups of adjacent peak frame indices that are each not more
    than `width` frames apart by the average of the frame indices.
    """
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)
