import org.python.util.PythonInterpreter;
import org.python.core.*;
import java.util.*;

public class myJavaClass{
    public static void main(String args[]) {
        PythonInterpreter interpreter = new PythonInterpreter();
        interpreter.execfile("./test.py");
		PyList result = (PyList)interpreter.eval("myPythonClass().abc(50)");
		for (int i = 0; i < 6; i++){
			PyObject obj = result.__getitem__(i);
			System.out.println(obj.asInt());
		}
		// System.out.println(list);
		// System.out.println((ArrayList) result);
    }
 }