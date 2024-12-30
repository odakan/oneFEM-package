##-----------------------------------------------------------------------##
#                                                                         #
#      #--oneFEM--#: One 2D FEM software in a galaxy far far away         #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         24 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#ALGORITHM main object definition
#   keeps universal vars, functions and list of analyses
#   if static solve u = K\F
#   if transient solve Ma + Cv + Ku = F

class Algorithm(object):
    def __init__(self):
        pass