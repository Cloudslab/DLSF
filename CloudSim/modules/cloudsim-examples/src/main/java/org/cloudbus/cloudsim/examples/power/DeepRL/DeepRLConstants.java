package org.cloudbus.cloudsim.examples.power.DeepRL;

import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.power.models.PowerModelSpecPowerDellPowerEdgeC6320;

/**
 * @author Shreshth Tuli
 * @since June 16, 2019
 */

public class DeepRLConstants {

    public final static String pythonCode = "../Deep-Learning/DeepRL.py";

    public final static String pythonClass = "DeepRL";

    public final static double meanGaussian = 6;

    public final static double stdGaussian = 5;

    public final static double vmTimemeanGaussian = 600; // 10 minutes

    public final static double vmTimestdGaussian = 300; // 5 minutes

    public final static int NUMBER_OF_VMS = 100;

    public final static long CLOUDLET_UTILIZATION_SEED = 1;

    public final static int NUMBER_OF_HOSTS = 100;


    public final static boolean ENABLE_OUTPUT = true;
    public final static boolean OUTPUT_CSV    = true; //

    public final static double SCHEDULING_INTERVAL = 300; // 5 min
    public final static double SIMULATION_LIMIT = 2*24*60*60; // 2 days

    public final static int CLOUDLET_LENGTH	=  5 * 60 * 1000; // 5 min
    public final static int CLOUDLET_PES	= 1;

    public  final  static  int NUMBER_OF_DATA_SAMPLES = (int) Math.ceil( SIMULATION_LIMIT / SCHEDULING_INTERVAL)+1; //  289;
    public  final  static double cpuUtilizationThreshold = 0.7;

    /**
     * VM instance types:Po
     *   High-Memory Extra Large Instance: 3.25 EC2 Compute Units, 8.55 GB // too much MIPS
     *   High-CPU Medium Instance: 2.5 EC2 Compute Units, 0.85 GB
     *   Extra Large Instance: 2 EC2 Compute Units, 3.75 GB
     *   Small Instance: 1 EC2 Compute Unit, 1.7 GB
     *   Micro Instance: 0.5 EC2 Compute Unit, 0.633 GB
     *   We decrease the memory size two times to enable oversubscription
     *
     **/
//    public final static int VM_TYPES	= 4;
//    public final static int[] VM_MIPS	= { 2500, 2000, 1000, 500 };
//    public final static int[] VM_PES	= { 1, 1, 1, 1 };
//    public final static int[] VM_RAM	= { 870,  1740, 1740, 613 };
//    public final static int VM_BW		= 100000; // 100 Mbit/s
//    public final static int VM_SIZE		= 2500; // 2.5 GB


    /** $ types of flavours selected from melbourne private research cloud
     *https://docs.cloud.unimelb.edu.au/guides/allocations/#premium-flavours
     * uom.general.1c4g	    1	4	Simple web hosting
     * uom.general.2c8g 	2	8	Database driven website
     * uom.general.4c16g	4	16	Data Science using RStudio or JupyterHub
     * uom.general.8c32g	8	32	Data science on larger data sets
     *  */
    public final static int VM_TYPES	= 4;
    public final static int[] VM_MIPS	= { 500, 1000, 2000, 2500 };
    public final static int[] VM_PES	= { 1, 2, 4, 8 };
    //Assumed 1 GB = 1000 MB in decimal
    public final static int[] VM_RAM	= { 4000, 8000, 16000, 32000};
    public final static int VM_BW		= 10000; // 10 Mbit/s
    public final static int VM_DISKBW  = 100000; // 100 Mbit/s
    public final static int VM_SIZE		= 2500; // 2.5 GB
    /*
     * Host types:
     *   HP ProLiant ML110 G4 (1 x [Xeon 3040 1860 MHz, 2 cores], 4GB)
     *   HP ProLiant ML110 G5 (1 x [Xeon 3075 2660 MHz, 2 cores], 4GB)
     *   We increase the memory size to enable over-subscription (x4)
     */

//
//    public final static int HOST_TYPES	 = 1;
//    public final static int[] HOST_MIPS	 = { 1860, 2660 }; //Original is 2660
//    public final static int[] HOST_PES	 = { 2, 2 };
//    public final static int[] HOST_RAM	 = { 4096, 4096 };
//    public final static int HOST_BW		 = 1000000; // 1 Gbit/s
//    public final static int HOST_STORAGE = 1000000; // 1 GB

    public final static int HOST_TYPES	 = 4; // 2
    public final static int[] HOST_MIPS	 = {2660, 2000, 2000, 2000};//{ 2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660,2660}; //Original is 2660
    public final static int[] HOST_PES	 = { 64, 32, 4, 2 }; // {2,2} /2
    public final static int[] HOST_RAM	 = { 64000, 48000, 16000, 8000 };// { 224096, 224096 };
    public final static int HOST_BW		 =  100000000; //  100 Gbits- > previous//1000000; // 1 Gbit/s
    public final static int HOST_DISK_BW    = 100000000;
    public final static int HOST_STORAGE = 1000000; // 1 GB


//
//    public final static int HOST_TYPES	 = 1;
//    public final static int[] HOST_MIPS	 = { 2660 }; //Original is 2660
//    public final static int[] HOST_PES	 = { 32 };
//    public final static int[] HOST_RAM	 = { 224096 };
//    public final static int HOST_BW		 = 10000000; // 1 Gbit/s // Bandwidth increase resulted in working, on less (0)bandwidth creates problem
//    public final static int HOST_STORAGE = 1000000; // 1 GB


    public final static PowerModel[] HOST_POWER = {
            new PowerModelSpecPowerDellPowerEdgeC6320()
//        new PowerModelSpecPowerIbmX3550XeonX5675()
    };

}
