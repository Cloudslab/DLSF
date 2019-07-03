package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.power.lists.PowerVmList;
import org.cloudbus.cloudsim.util.ExecutionTimeMeasurer;

import java.util.*;

public class DRLVmAllocationPolicy extends PowerVmAllocationPolicyAbstract{

    /** The vm selection policy. */
    private PowerVmSelectionPolicy vmSelectionPolicy;

    /** A list of maps between a VM and the host where it is place.
     * @todo This list of map is implemented in the worst way.
     * It should be used just a Map<Vm, Host> to find out
     * what PM is hosting a given VM.
     */
    private final List<Map<String, Object>> savedAllocation = new ArrayList<Map<String, Object>>();

    /** A map of CPU utilization history (in percentage) for each host,
     where each key is a host id and each value is the CPU utilization percentage history.*/
    private final Map<Integer, List<Double>> utilizationHistory = new HashMap<Integer, List<Double>>();

    /** The history of time spent in VM reallocation
     * every time the optimization of VM allocation method is called.
     * @see #optimizeAllocation(java.util.List)
     */
    private final List<Double> executionTimeHistoryVmReallocation = new LinkedList<Double>();

    /** The history of total time spent in every call of the
     * optimization of VM allocation method.
     * @see #optimizeAllocation(java.util.List)
     */
    private final List<Double> executionTimeHistoryTotal = new LinkedList<Double>();

    /** The fallback VM allocation policy to be used when
     * the IQR over utilization host detection doesn't have
     * data to be computed. */
    private PowerVmAllocationPolicyMigrationAbstract fallbackVmAllocationPolicy;

    /** Host Penalty =
     * The number of hosts that came in top rank by DL result which were not suitable
     */
    public int hostPenalty = 0;

    /**
     * Instantiates a new DRLVmAllocationPolicy
     *
     * @param hostList the host list
     */
    public DRLVmAllocationPolicy(
            List<? extends Host> hostList,
            DRLVmSelectionPolicy selectionPolicy,
            PowerVmAllocationPolicyMigrationAbstract fallbackVmAllocationPolicy) {
        super(hostList);
        this.vmSelectionPolicy = selectionPolicy;
        setFallbackVmAllocationPolicy(fallbackVmAllocationPolicy);
        hostPenalty = 0;
    }

    /**
     * The method doesn't perform any VM allocation optimization
     * and in fact has no effect.
     * @param vmList
     * @return
     */
    @Override
    public List<Map<String, Object>> optimizeAllocation(List<? extends Vm> vmList) {
        List<? extends Vm> vmsToMigrate = this.vmSelectionPolicy.getAllVmsToMigrate(this.getHostList(), vmList);
        saveAllocation();
        ExecutionTimeMeasurer.start("optimizeAllocationVmReallocation");
        List<Map<String, Object>> migrationMap = getNewVmPlacement(vmsToMigrate, new HashSet<Host>(
                this.getHostList()), vmList);
        getExecutionTimeHistoryVmReallocation().add(
                ExecutionTimeMeasurer.end("optimizeAllocationVmReallocation"));
        Log.printLine();
        restoreAllocation();

//        getExecutionTimeHistoryTotal().add(ExecutionTimeMeasurer.end("optimizeAllocationTotal"));

        return migrationMap;
    }

    /**
     * Gets a new vm placement considering the list of VM to migrate.
     *
     * @param vmsToMigrate the list of VMs to migrate
     * @param excludedHosts the list of hosts that aren't selected as destination hosts
     * @return the new vm placement map
     */
    protected List<Map<String, Object>> getNewVmPlacement(
            List<? extends Vm> vmsToMigrate,
            Set<? extends Host> excludedHosts,
            List<? extends Vm> vmList) {
        List<Map<String, Object>> migrationMap = new LinkedList<Map<String, Object>>();
        PowerVmList.sortByCpuUtilization(vmsToMigrate);
        this.hostPenalty = 0;
        int vmIndex = 0;
        for (Vm vm : vmsToMigrate) {
            PowerHost allocatedHost = this.<PowerHost>getHostList().get(1);
            // @todo allocatedHost = parse from DL output
            vmIndex = vmList.indexOf(vm);
            String result; ArrayList<String> sortedHosts = new ArrayList();
            try{
                DRLDatacenter.toPython.println(("getSortedHost\n"+vmIndex+"\nEND")); DRLDatacenter.toPython.flush();
                result = DRLDatacenter.fromPython.readLine();
                sortedHosts = new ArrayList<>(Arrays.asList(result.split(" ")));
            }
            catch(Exception e){
                System.out.println(e.getMessage());
            }
            for(int i = 0; i < this.getHostList().size(); i++){
                allocatedHost = this.<PowerHost>getHostList().get(Integer.parseInt(sortedHosts.get(i)));
                if(allocatedHost.isSuitableForVm(vm)){
                    break;
                }
                hostPenalty += 1;
            }
            if (allocatedHost != null) {
                allocatedHost.vmCreate(vm);
                Log.printConcatLine("VM #", vm.getId(), " allocated to host #", allocatedHost.getId());

                Map<String, Object> migrate = new HashMap<String, Object>();
                migrate.put("vm", vm);
                migrate.put("host", allocatedHost);
                migrationMap.add(migrate);
            }
        }
        return migrationMap;
    }

    /**
     * Updates the list of maps between a VM and the host where it is place.
     * @see #savedAllocation
     */
    protected void saveAllocation() {
        getSavedAllocation().clear();
        for (Host host : getHostList()) {
            for (Vm vm : host.getVmList()) {
                if (host.getVmsMigratingIn().contains(vm)) {
                    continue;
                }
                Map<String, Object> map = new HashMap<String, Object>();
                map.put("host", host);
                map.put("vm", vm);
                getSavedAllocation().add(map);
            }
        }
    }

    /**
     * Restore VM allocation from the allocation history.
     * @see #savedAllocation
     */
    protected void restoreAllocation() {
        for (Host host : getHostList()) {
            host.vmDestroyAll();
            host.reallocateMigratingInVms();
        }
        for (Map<String, Object> map : getSavedAllocation()) {
            Vm vm = (Vm) map.get("vm");
            PowerHost host = (PowerHost) map.get("host");
            if (!host.vmCreate(vm)) {
                Log.printConcatLine("Couldn't restore VM #", vm.getId(), " on host #", host.getId());
                System.exit(0);
            }
            getVmTable().put(vm.getUid(), host);
        }
    }

    /**
     * Gets the utilization history.
     *
     * @return the utilization history
     */
    public Map<Integer, List<Double>> getUtilizationHistory() {
        return utilizationHistory;
    }

    /**
     * Gets the saved allocation.
     *
     * @return the saved allocation
     */
    protected List<Map<String, Object>> getSavedAllocation() {
        return savedAllocation;
    }

    /**
     * Gets the execution time history vm reallocation.
     *
     * @return the execution time history vm reallocation
     */
    public List<Double> getExecutionTimeHistoryVmReallocation() {
        return executionTimeHistoryVmReallocation;
    }

    /**
     * Gets the execution time history total.
     *
     * @return the execution time history total
     */
    public List<Double> getExecutionTimeHistoryTotal() {
        return executionTimeHistoryTotal;
    }

    /**
     * Sets the fallback vm allocation policy.
     *
     * @param fallbackVmAllocationPolicy the new fallback vm allocation policy
     */
    public void setFallbackVmAllocationPolicy(
            PowerVmAllocationPolicyMigrationAbstract fallbackVmAllocationPolicy) {
        this.fallbackVmAllocationPolicy = fallbackVmAllocationPolicy;
    }

    /**
     * Gets the fallback vm allocation policy.
     *
     * @return the fallback vm allocation policy
     */
    public PowerVmAllocationPolicyMigrationAbstract getFallbackVmAllocationPolicy() {
        return fallbackVmAllocationPolicy;
    }

    /**
     * Sets the vm selection policy.
     *
     * @param vmSelectionPolicy the new vm selection policy
     */
    protected void setVmSelectionPolicy(PowerVmSelectionPolicy vmSelectionPolicy) {
        this.vmSelectionPolicy = vmSelectionPolicy;
    }

    /**
     * Gets the vm selection policy.
     *
     * @return the vm selection policy
     */
    protected PowerVmSelectionPolicy getVmSelectionPolicy() {
        return vmSelectionPolicy;
    }

}
