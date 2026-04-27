from stable_datasets.schema import ClassLabel, Sequence
from stable_datasets.timeseries import (
    AudioMNIST,
    BirdVoxDCASE20k,
    Freefield1010,
    GrooveMIDI,
    GTZAN,
    Picidae,
    SeizuresNeonatal,
    SONYCUST,
    SpeechCommands,
    VoiceGenderDetection,
    VocalSet,
    Warblr,
)


def _builder_info(builder_cls):
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder)
    return builder.info


def test_wave1_timeseries_builders_have_series_and_classlabel():
    for builder_cls in [
        AudioMNIST,
        BirdVoxDCASE20k,
        Freefield1010,
        GTZAN,
        Picidae,
        SpeechCommands,
        VoiceGenderDetection,
        Warblr,
    ]:
        info = _builder_info(builder_cls)
        assert "series" in info.features
        assert isinstance(info.features["series"], Sequence)
        assert "label" in info.features
        assert isinstance(info.features["label"], ClassLabel)


def test_structured_timeseries_builders_have_series_feature():
    for builder_cls in [GrooveMIDI, SeizuresNeonatal, SONYCUST, VocalSet]:
        info = _builder_info(builder_cls)
        assert "series" in info.features
        assert isinstance(info.features["series"], Sequence)
