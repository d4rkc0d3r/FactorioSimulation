using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FactorioSimulationStuff
{
    class Program
    {
        static void Main(string[] args)
        {
            var vanillaModules = new[] {
                new {name = "productivity 1", speed = -0.15m, energy = 0.4m, productivity = 0.04m},
                new {name = "productivity 2", speed = -0.15m, energy = 0.6m, productivity = 0.06m},
                new {name = "productivity 3", speed = -0.15m, energy = 0.8m, productivity = 0.1m},
                new {name = "efficiency 1", speed = 0.0m, energy = -0.3m, productivity = 0.0m},
                new {name = "efficiency 2", speed = 0.0m, energy = -0.4m, productivity = 0.0m},
                new {name = "efficiency 3", speed = 0.0m, energy = -0.5m, productivity = 0.0m},
                new {name = "speed 1", speed = 0.2m, energy = 0.5m, productivity = 0.0m},
                new {name = "speed 2", speed = 0.3m, energy = 0.6m, productivity = 0.0m},
                new {name = "speed 3", speed = 0.5m, energy = 0.7m, productivity = 0.0m}
            };

            var bobModules = new[] {
                new {name = "speed 1", speed = 0.20m, energy = 0.15m, productivity = 0.0m}
            }.ToList();

            bobModules.Clear();

            for (int i = 1; i <= 4; i++)
            {
                bobModules.Add(new { name = "speed " + i, speed = 0.20m * i, energy = 0.15m * i, productivity = 0.0m });
                bobModules.Add(new { name = "effectivity " + i, speed = 0.0m, energy = -0.15m * i, productivity = 0.0m });
                bobModules.Add(new { name = "productivity " + i, speed = -0.20m + -0.05m * i, energy = 0.15m * i, productivity = 0.05m * i });
            }

            var modules = vanillaModules;

            foreach (var m in modules)
            {
                Console.WriteLine(m);
            }

            Console.WriteLine();

            int moduleSlots = 4;
            var moduleIndex = new int[moduleSlots];
            for (int i = 0; i < moduleSlots; i++)
            {
                moduleIndex[i] = 0;
            }

            var combinations = new[] {
                new {name = "none", productivity = 1.0m, energyPerResult = 1.0m}
            }.ToList();
            combinations.Clear();

            while (moduleIndex[0] < modules.Count())
            {
                string name = "";
                decimal energy = 1;
                decimal speed = 1;
                decimal productivity = 1;

                for (int i = 0; i < moduleIndex.Count(); i++)
                {
                    var m = modules[moduleIndex[i]];
                    energy += m.energy;
                    speed += m.speed;
                    productivity += m.productivity;
                    name += m.name[0];
                    name += m.name[m.name.Count() - 1];
                }

                energy = (energy < 0.2m) ? 0.2m : energy;
                speed = (speed < 0.2m) ? 0.2m : speed;

                decimal energyPerResult = (energy / speed) / productivity;

                combinations.Add(new { name = name, productivity = productivity, energyPerResult = energyPerResult });

                Console.WriteLine(combinations.Last());

                if (++moduleIndex[moduleIndex.Count() - 1] == modules.Count())
                {
                    int i = moduleIndex.Count() - 2;
                    for (; i >= 0; i--)
                    {
                        if (++moduleIndex[i] < modules.Count())
                        {
                            break;
                        }
                    }
                    for (; i < moduleIndex.Count() - 1; i++)
                    {
                        if (i > -1)
                        {
                            moduleIndex[i + 1] = moduleIndex[i];
                        }
                    }
                }
            }

            Console.WriteLine();

            var l = combinations.Select(a => a.productivity).Distinct().Select(a => combinations.Where(b => b.productivity == a).OrderBy(b => b.energyPerResult).First()).OrderBy(a => a.energyPerResult).ToList();

            var last = l.First();

            foreach (var a in l)
            {
                if (a == last || a.productivity > last.productivity)
                {
                    Console.WriteLine(a);
                    last = a;
                }
            }

            Console.WriteLine();

            Console.WriteLine("Press any key to close the application");
            Console.ReadKey();
        }
    }
}
