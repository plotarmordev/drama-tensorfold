"""The winner among the smaller builds: the smallest whose word errors are not doubled and whose failed takes are
not above the baseline's (t5900-cover) in either voice."""


def qualifies(build, base):
    """True when every voice of `build` keeps mean WER at most twice the baseline's and no more failed takes."""
    return all(build['voices'][voice]['wer_mean'] <= 2 * score['wer_mean']
               and build['voices'][voice]['failed'] <= score['failed'] for voice, score in base['voices'].items())


def winner(builds, base_name):
    """The name of the smallest qualifying build; `builds` maps name -> {'bytes': n, 'voices': {voice: score}}."""
    base = builds[base_name]
    return min((name for name, build in builds.items() if qualifies(build, base)), key=lambda n: builds[n]['bytes'])
