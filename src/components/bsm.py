import math
import numpy
from abc import abstractmethod

from .photon import Photon
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..utils.encoding import time_bin
from ..utils.quantum_state import QuantumState


def make_bsm(name, timeline, **kwargs):
    encoding_type == kwargs.get("encoding_type", "time_bin")

    if encoding_type == "polarization":
        return PolarizationBSM(name, timeline, **kwargs)
    elif encoding_type == "time_bin":
        return TimeBinBSM(name, timeline, **kwargs)
    elif encoding_type == "ensemble":
        return EnsembleBSM(name, timeline, **kwargs)
    elif encoding_type == "single_atom":
        return SingleAtomBSM(name, timeline, **kwargs)
    else:
        raise Exception("invalid encoding {} given for BSM {}".format(encoding_type, name))


# abstract parent BSM class
class BSM(Entity):
    def __init__(self, name, timeline, **kwargs):
        super().__init__(name, timeline)
        self.encoding_type = kwargs.get("encoding_type", time_bin)
        self.phase_error = kwargs.get("phase_error", 0)
        self.photons = []
        self.photon_arrival_time = -1
        detectors = kwargs.get("detectors", [])

        self.detectors = []
        for d in detectors:
            if d is not None:
                detector = Detector(timeline, **d)
                detector.parents.append(self)
            else:
                detector = None
            self.detectors.append(detector)

        # get resolution
        self.resolution = max(d.time_resolution for d in self.detectors)

        # define bell basis vectors
        self.bell_basis = [[complex(math.sqrt(1 / 2)), complex(0), complex(0), complex(math.sqrt(1 / 2))],
                           [complex(math.sqrt(1 / 2)), complex(0), complex(0), -complex(math.sqrt(1 / 2))],
                           [complex(0), complex(math.sqrt(1 / 2)), complex(math.sqrt(1 / 2)), complex(0)],
                           [complex(0), complex(math.sqrt(1 / 2)), -complex(math.sqrt(1 / 2)), complex(0)]]

    def init(self):
        for detector in self.detectors:
            detector.init()

    @abstractmethod
    def get(self, photon):
        # check if photon arrived later than current photon
        if self.photon_arrival_time < self.timeline.now():
            # clear photons
            for old_photon in self.photons:
                old_photon.remove_from_timeline()
            self.photons = [photon]
            # set arrival time
            self.photon_arrival_time = self.timeline.now()

        # check if we have a photon from a new location
        if not any([reference.location == photon.location for reference in self.photons]):
            self.photons.append(photon)

    @abstractmethod
    def pop(self, **kwargs):
        # calculate bsm based on detector num
        detector = kwargs.get("detector")
        detector_num = self.detectors.index(detector)
        time = kwargs.get("time")


class PolarizationBSM(BSM):
    def __init__(self, name, timeline, **kwargs):
        super().__init__(name, timeline, **kwargs)
        assert len(self.detectors) == 4

    def get(self, photon):
        super().get(self, photon)
        # TODO

    def pop(self, **kwargs):
        # TODO
        pass


def TimeBinBSM(BSM):
    def __init__(self, name, timeline, **kwargs):
        super().__init__(name, timeline, **kwargs)
        assert len(self.detectors) == 2

    def get(self, photon):
        super().get(photon)

        if len(self.photons) != 2:
            return

        if numpy.random.random_sample() < self.phase_error:
            self.photons[1].apply_phase_error()
        # entangle photons to measure
        self.photons[0].entangle(self.photons[1])

        # measure in bell basis
        res = Photon.measure_multiple(self.bell_basis, self.photons)

        # check if we've measured as Phi+ or Phi-; these cannot be measured by the BSM
        if res == 0 or res == 1:
            return

        early_time = self.timeline.now()
        late_time = early_time + self.encoding_type["bin_separation"]

        # measured as Psi+
        # send both photons to the same detector at the early and late time
        if res == 2:
            detector_num = numpy.random.choice([0, 1])

            process = Process(self.detectors[detector_num], "get", [])
            event = Event(int(round(early_time)), process)
            self.timeline.schedule(event)
            process = Process(self.detectors[detector_num], "get", [])
            event = Event(int(round(late_time)), process)
            self.timeline.schedule(event)

        # measured as Psi-
        # send photons to different detectors at the early and late time
        elif res == 3:
            detector_num = numpy.random.choice([0, 1])

            process = Process(self.detectors[detector_num], "get", [])
            event = Event(int(round(early_time)), process)
            self.timeline.schedule(event)
            process = Process(self.detectors[1 - detector_num], "get", [])
            event = Event(int(round(late_time)), process)
            self.timeline.schedule(event)

        # invalid result from measurement
        else:
            raise Exception("Invalid result from photon.measure_multiple")

    def pop(self, **kwargs):
        # TODO
        pass


def EnsembleBSM(BSM):
    def __init__(self, name, timeline, **kwargs):
        super().__init__(name, timeline, **kwargs)
        self.previous_state = None
        assert len(self.detectors) == 2

    def get(self, photon):
        super().get(photon)

        # TODO: what if photon lose in channel
        if len(self.photons) == 1:
            return
        elif len(self.photons) == 2:
            mem_0 = self.photons[0].encoding_type["memory"]
            mem_1 = self.photons[1].encoding_type["memory"]

            is_valid = self.photons[0].is_null ^ self.photons[1].is_null

            # if we have 1 photon, generate entanglement
            if is_valid:
                qstate_0 = mem_0.qstate
                qstate_1 = mem_1.qstate
                # if unentangled, entangle
                if qstate_0 not in qstate_1.entangled_states:
                    qstate_0.entangle(qstate_1)
                self.previous_state = mem_0.qstate.state
                # project to bell basis
                _ = QuantumState.measure_multiple(self.bell_basis, [qstate_0, qstate_1])
                # send detect message to a random detector
                detector_num = numpy.random.choice([0, 1])
                self.detectors[detector_num].get()

            # if we have 2 photons, have both detectors get
            elif not self.photons[0].is_null:
                self.detectors[0].get()
                self.detectors[1].get()

    def pop(self, **kwargs):
        # calculate bsm based on detector num
        detector = kwargs.get("detector")
        detector_num = self.detectors.index(detector)
        time = kwargs.get("time")

        res = detector_num
        self._pop(entity="BSM", res=res, time=time)


class SingleAtomBSM(BSM):
    def __init__(self, name, timeline, **kwargs):
        super().__init__(name, timeline, **kwargs)
        self.second_round = False
        assert len(self.detectors) == 4

    def get(self, photon):
        super().get(photon)

        memory = photon.encoding_type["memory"]

        # check if we're in first stage. If we are and not null, send photon to random detector
        if memory.previous_bsm == -1 and not photon.is_null:
            detector_num = numpy.random.choice([0, 1])
            memory.previous_bsm = detector_num
            self.detectors[detector_num].get()

        if len(self.photons) == 2:
            null_0 = self.photons[0].is_null
            null_1 = self.photons[1].is_null
            is_valid = null_0 ^ null_1
            if is_valid:
                memory_0 = self.photons[0].encoding_type["memory"]
                memory_1 = self.photons[1].encoding_type["memory"]
                # if we're in stage 1: null photon will need bsm assigned
                if null_0 and memory_0.previous_bsm == -1:
                    memory_0.previous_bsm = memory_1.previous_bsm
                elif null_1 and memory_1.previous_bsm == -1:
                    memory_1.previous_bsm = memory_0.previous_bsm
                # if we're in stage 2: send photon to same (opposite) detector for psi+ (psi-)
                else:
                    if memory_0.qstate not in memory_1.qstate.entangled_states:
                        memory_0.qstate.entangle(memory_1.qstate)
                    res = QuantumState.measure_multiple(self.bell_basis, [memory_0.qstate, memory_1.qstate])
                    if res == 2:  # Psi+
                        detector_num = memory_0.previous_bsm
                    elif res == 3:  # Psi-
                        detector_num = 1 - memory_0.previous_bsm
                    else:
                        raise Exception("invalid bell state result {}".format(res))
                    self.detectors[detector_num].get()

    def pop(self, **kwargs):
        detector = kwargs.get("detector")
        detector_num = self.detectors.index(detector)
        time = kwargs.get("time")

        res = detector_num
        self._pop(entity="BSM", res=res, time=time)


