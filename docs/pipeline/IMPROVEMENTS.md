***This Document is to house improvement ideas for our live inference pipeline***

One thing we absolutely need is a way to only run inference on games that have a meaningful delta. Currently we run all games every time we run the pipeline (expensive) We should only run inference when a game context meaningfully changes. A player ruled out, a different starting lineup than projected is confirmed, a large line / player prop shift. Things like that.

Not sure we handle rotowire projected --> confirmed starters the best way

Injury pipeline seems solid

other jobs -- eval seems serviceable for now, box scores, i think we scrape them but we don't do anything. i'd like to get set up for automated model re-training. not sure what all that en-tails

