using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;

namespace TIS_100_Compiler
{
    class Program
    {
        static void Main(string[] args)
        {
            string tamplateName = "TIS-100-ROM16-template";
            string bpFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                                         "factorio\\script-output\\blueprint-string\\" + tamplateName + ".txt");

            string compressedBlueprint = File.ReadAllText(bpFile);

            string blueprint = Unzip(Convert.FromBase64String(compressedBlueprint));

            string[] source = File.ReadAllLines("sampleCode.txt");

            source = source.Where(line => !line.StartsWith("#")).ToArray();

            int[] lineToInstructionMap = new int[source.Length];
            Instruction[] code = new Instruction[16];
            for(int i = 0; i < 16; i++)
            {
                code[i] = new Instruction(i + 1);
            }
            Dictionary<string, int> labels = new Dictionary<string,int>();

            // first pass, building maps
            int highestInstruction = 0;
            for(int i = 0; i < source.Length; i++)
            {
                source[i] = source[i].Trim().ToUpperInvariant();
                lineToInstructionMap[i] = highestInstruction;
                if (source[i].Contains(":"))
                {
                    string label = source[i].Split(new char[] {':'}, 2)[0];
                    if (!label.All(c => Char.IsLetter(c)))
                    {
                        //TODO: syntax error handling
                    }
                    else
                    {
                        labels.Add(label, highestInstruction);
                    }
                }
                if (!source[i].EndsWith(":"))
                {
                    highestInstruction++;
                }
            }

            // second pass, compiling instructions
            for (int i = 0; i < source.Length; i++)
            {
                string tmp = source[i];
                if(source[i].Contains(":"))
                {
                    tmp = source[i].Split(new char[] { ':' }, 2)[1].TrimStart();
                }
                if(code.Equals(""))
                {
                    continue;
                }
                string[] parameter;
                string instruction;
                if (tmp.Contains(' '))
                {
                    string[] split = tmp.Split(new char[] { ' ' }, 2);
                    instruction = split[0];
                    parameter = split[1].Split(new char[] { ',' }).Select(p => p.Trim()).ToArray();
                }
                else
                {
                    parameter = new string[0];
                    instruction = tmp;
                }
                Instruction inst = code[lineToInstructionMap[i]];
                int nextInstruction = ((lineToInstructionMap[i] + 1) % highestInstruction) + 1;
                int number;
                switch(instruction)
                {
                    case "MOV":
                        if(parameter[1].Equals("ACC"))
                        {
                            inst.Add("1", nextInstruction);
                        }
                        if(int.TryParse(parameter[0], out number))
                        {
                            if(parameter[1].Equals("ACC"))
                            {
                                inst.Add("V", number);
                                inst.Add("R", -1);
                            }
                            else
                            {

                            }
                        }
                        else
                        {

                        }
                        break;
                    case "ADD":
                        if (int.TryParse(parameter[0], out number))
                        {
                            inst.Add("1", nextInstruction);
                            inst.Add("V", number);
                        }
                        else
                        {
                            if (parameter[0].Equals("ACC"))
                            {
                                inst.Add("1", nextInstruction);
                                inst.Add("M", 1);
                            }
                            else
                            {

                            }
                        }
                        break;
                    case "SUB":
                        if (int.TryParse(parameter[0], out number))
                        {
                            inst.Add("1", nextInstruction);
                            inst.Add("V", -number);
                        }
                        else
                        {

                        }
                        break;
                    case "NEG":
                        inst.Add("1", nextInstruction);
                        inst.Add("M", -2);
                        break;
                    case "MUL":
                        if (int.TryParse(parameter[0], out number))
                        {
                            inst.Add("1", nextInstruction);
                            inst.Add("M", number);
                            inst.Add("R", -1);
                        }
                        else
                        {
                            if (parameter[0].Equals("ACC"))
                            {
                                inst.Add("2", nextInstruction);
                                inst.Add("W", 1);
                                inst.Add("R", -1);
                            }
                            else
                            {

                            }
                        }
                        break;
                    case "DIV":
                        if (int.TryParse(parameter[0], out number))
                        {
                            inst.Add("1", nextInstruction);
                            inst.Add("D", number);
                            inst.Add("R", -1);
                        }
                        else
                        {
                            
                        }
                        break;
                    case "SAV":
                        inst.Add("1", nextInstruction);
                        inst.Add("Q", -1);
                        inst.Add("B", 1);
                        break;
                    case "SWP":
                        inst.Add("1", nextInstruction);
                        inst.Add("Q", -1);
                        inst.Add("B", 1);
                        inst.Add("R", -1);
                        inst.Add("L", 1);
                        break;
                    case "NOP":
                        inst.Add("1", nextInstruction);
                        break;
                    case "JMP":
                        inst.Add("1", labels[parameter[0]] + 1);
                        break;
                    case "JEZ":
                        inst.Add("1", nextInstruction);
                        inst.AddItem("pistol", labels[parameter[0]] + 1 - nextInstruction);
                        break;
                    case "JNZ":
                        inst.Add("1", nextInstruction);
                        inst.AddItem("combat-shotgun", labels[parameter[0]] + 1 - nextInstruction);
                        break;
                    case "JGZ":
                        inst.Add("1", nextInstruction);
                        inst.AddItem("submachine-gun", labels[parameter[0]] + 1 - nextInstruction);
                        break;
                    case "JLZ":
                        inst.Add("1", nextInstruction);
                        inst.AddItem("shotgun", labels[parameter[0]] + 1 - nextInstruction);
                        break;
                    case "HLT":
                        break;
                    default:
                        if (instruction.GetHashCode() == -1289182540)
                        {
                            inst.AddVirtual("red", 1);
                        }
                        else
                        {

                        }
                        break;
                }
            }
            
            for(int i = 0; i < code.Length; i++)
            {
                blueprint = code[i].PlaceInBlueprint(blueprint);
            }

            File.WriteAllText("blueprint.txt", Convert.ToBase64String(Zip(blueprint)));
        }

        public static void CopyTo(Stream src, Stream dest)
        {
            byte[] bytes = new byte[4096];

            int cnt;

            while ((cnt = src.Read(bytes, 0, bytes.Length)) != 0)
            {
                dest.Write(bytes, 0, cnt);
            }
        }

        public static byte[] Zip(string str)
        {
            var bytes = Encoding.UTF8.GetBytes(str);

            using (var msi = new MemoryStream(bytes))
            using (var mso = new MemoryStream())
            {
                using (var gs = new GZipStream(mso, CompressionMode.Compress))
                {
                    CopyTo(msi, gs);
                }

                return mso.ToArray();
            }
        }

        public static string Unzip(byte[] bytes)
        {
            using (var msi = new MemoryStream(bytes))
            using (var mso = new MemoryStream())
            {
                using (var gs = new GZipStream(msi, CompressionMode.Decompress))
                {
                    CopyTo(gs, mso);
                }

                return Encoding.UTF8.GetString(mso.ToArray());
            }
        }
    }
}
