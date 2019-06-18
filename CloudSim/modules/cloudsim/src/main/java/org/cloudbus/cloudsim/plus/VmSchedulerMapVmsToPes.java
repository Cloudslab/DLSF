package org.cloudbus.cloudsim.plus;

import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Vm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 
 * A scheduler, which maps vms to pes.
 * 
 * @author nikolay.grozev
 * 
 */
public abstract class VmSchedulerMapVmsToPes<P extends Pe> extends VmSchedulerWithIndependentPes<P> {

    private final Map<Integer, List<Integer>> vmsToPes = new HashMap<>();

    public VmSchedulerMapVmsToPes(final List<P> pelist) {
        super(pelist);
    }

    /**
     * Maps the vm with the specified id to the pe, specified by its id.
     * 
     * @param vmid
     *            - the id of the vm.
     * @param peid
     *            - the id of the pe.
     */
    public void map(final int vmid, final int peid) {
        if (vmsToPes.containsKey(vmid)) {
            vmsToPes.get(vmid).add(peid);
        } else {
            vmsToPes.put(vmid, new ArrayList<Integer>());
            vmsToPes.get(vmid).add(peid);
        }
    }

    @Override
    public boolean doesVmUse(final Vm vm, final Pe pe) {
        return vmsToPes.containsKey(vm.getId()) && vmsToPes.get(vm.getId()).contains(pe.getId());
    }

}
