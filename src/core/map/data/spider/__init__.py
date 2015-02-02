# -----------------------------------------------------------------------------
# SPIDER file reader.
# Used by EM data processing program SPIDER.
#
def open(path):

  from spider_grid import SPIDER_Grid
  return [SPIDER_Grid(path)]
