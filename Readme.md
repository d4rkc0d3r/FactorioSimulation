# Command line belt balancer analyzer

Copyright d4rkpl4y3r <real name here> 2016-2017

Licensed under <License Type Here, I recommend MIT or BSD>

Using code from: <if you have any code from someone else put it here.  THIS MAY FORCE YOUR LICENSING CHOICE>


When designing belt balancers it usually takes a large amount of time to properly test it. Therefore I decided to write a tool to help me with that. No more need to painfully test every input ingame. Now I also decided to release it to the public.

You can get it from [github](https://github.com/d4rkc0d3r/FactorioSimulation/releases).

Because of technical reasons the balancer that is tested needs to have each input and output as a belt piece. For example this 12 to 12 balancer:


![Image](http://i.imgur.com/DmxRel3.png?1)


Also side loading is a no go.

# There are some commandline options that I will explain:

* -f=FILE Loads the blueprint string file FILE, if not found tries again with %APPDATA%\factorio\script-output\blueprint-string\FILE
* -t2 Tests all throughput combinations where exactly two inputs and outputs are used.
* -tallcpu Tests all throughput combinations where more or equal to two inputs and outputs are used. (The 12 belt balancer needed 10 minutes on my pc with this option)
* -i=N Specifies the number of iterations the simulation should run. Default is 2 * (2 * nSplitter + nInputs + nOutputs + 1)
* -time Times the complete testing time needed.
* -s Does suppress the ongoing progress display. Useful if you pipe the output to a file.
* -benchmark times only the simulation time needed to run the specified amount of iterations.

-f=FILE is mandatory. Then there is also -gpu using CUDA, but the overhead is usually larger than the performance gain. There was one case with the big 512 belt balancer where this option did actually win against the cpu, but for all sane balancers (<= 32 belts), cpu wins. -tallgpu does also exist, but it crashes my display driver more often than not.

I usually have a cmd window open with the line:
`BeltBalancer.exe -f=test.txt -t2`

Then I can just save my balancer I am working on as test, swap to the console hit the up arrow and enter.

# Forum Link:

See [this](https://forums.factorio.com/viewtopic.php?f=69&t=34182) thread for more information.
