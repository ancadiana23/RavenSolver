py -3 main\nn_agent.py --data imgs --ex memorize --nn conv > results\nn_agent_imgs_memorize_conv.txt
py -3 main\nn_agent.py --data imgs --ex solve --nn conv > results\nn_agent_imgs_solve_conv.txt
py -3 main\nn_agent.py --data sdrs --ex memorize --nn conv > results\nn_agent_sdrs_memorize_conv.txt
py -3 main\nn_agent.py --data sdrs --ex solve --nn conv > results\nn_agent_sdrs_solve_conv.txt
py -3 main\nn_agent.py --data symbolic --ex solve --nn conv > results\nn_agent_symbolic_solve_conv.txt
