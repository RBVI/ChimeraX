import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative').print_stats()
