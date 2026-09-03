TEST_POSITIONS = {
    # 1. Starting position — baseline branching factor
    "starting": (
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ),

    # 2. Open middlegame — lots of legal moves
    "open_middlegame": (
        "r2q1rk1/ppp2ppp/2np1n2/8/2B1P3/"
        "2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1"
    ),


    # 5. Endgame — few pieces / low branching factor
    "endgame": (
        "8/5pk1/3p2p1/2pP4/2P2P2/"
        "5K2/6PP/8 w - - 0 1"
    ),

    # 9. Almost no tactical activity — positional
    "closed": (
        "r1bqk2r/pp1n1ppp/2p1pn2/3p4/"
        "3P4/2N1PN2/PPP2PPP/R1BQKB1R w KQkq - 0 1"
    ),

    # 10. Many pieces + tactical opportunities
    "complex": (
        "r3k2r/ppp1qppp/2n1b3/3p4/"
        "3P4/2N1PN2/PPP1QPPP/R3KB1R w KQkq - 0 1"
    ),

}