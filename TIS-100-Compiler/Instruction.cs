using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TIS_100_Compiler
{
    class Instruction
    {
        private Dictionary<String, Int32> signalDictionary = new Dictionary<string,int>();
        private int address;

        public Instruction(int address)
        {
            this.address = address;
        }

        public void Add(String signal, int value)
        {
            if (signal.Length == 1)
            {
                signal = "{type=\"virtual\",name=\"signal-" + signal.ToUpperInvariant() + "\"}";
            }
            signalDictionary.Add(signal, value);
        }

        public void AddVirtual(String signal, int value)
        {
            signal = "{type=\"virtual\",name=\"signal-" + signal + "\"}";
            signalDictionary.Add(signal, value);
        }

        public void AddItem(String signal, int value)
        {
            signal = "{type=\"item\",name=\"" + signal + "\"}";
            signalDictionary.Add(signal, value);
        }

        public string PlaceInBlueprint(string blueprint)
        {
            string toReplace = "{signal={type=\"virtual\",name=\"signal-A\"},count=" + address + ",index=1}";
            string newValue = "";
            int index = 1;
            foreach(string signal in signalDictionary.Keys)
            {
                newValue += "{signal=" + signal + ",count=" + ((UInt32)signalDictionary[signal]) + ",index=" + index + "},";
                index++;
            }
            if(newValue.Length > 0)
            {
                newValue = newValue.Substring(0, newValue.Length - 1); // trim last ,
            }
            return blueprint.Replace(toReplace, newValue);
        }
    }
}
