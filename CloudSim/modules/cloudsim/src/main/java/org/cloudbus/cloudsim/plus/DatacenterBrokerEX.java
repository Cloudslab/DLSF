package org.cloudbus.cloudsim.plus;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.plus.billing.IVmBillingPolicy;
import org.cloudbus.cloudsim.plus.util.CustomLog;
import org.cloudbus.cloudsim.plus.vm.VMStatus;
import org.cloudbus.cloudsim.plus.vm.VMex;
import org.cloudbus.cloudsim.lists.VmList;

import java.math.BigDecimal;
import java.util.*;
import java.util.logging.Level;

/**
 * 
 * An extension to the default broker, which allows (i) VM destruction; (ii)
 * definition of events prior to simulation start, which are fired later on;
 * (iii) predefined "life length" of the broker; (iv) billing.
 * 
 * @author nikolay.grozev
 * 
 */
public class DatacenterBrokerEX extends DatacenterBroker {

    // FIXME find a better way to get an unused tag instead of hardcoding 123456
    protected static final int BROKER_DESTROY_ITSELF_NOW = 123456;
    protected static final int BROKER_DESTROY_VMS_NOW = BROKER_DESTROY_ITSELF_NOW + 1;
    protected static final int BROKER_SUBMIT_VMS_NOW = BROKER_DESTROY_ITSELF_NOW + 2;
    protected static final int BROKER_CLOUDLETS_NOW = BROKER_DESTROY_ITSELF_NOW + 3;

    /** Number of VM destructions requested. */
    private int vmDestructsRequested = 0;

    /** Number of VM destructions acknowledged. */
    private int vmDestructsAcks = 0;

    /**
     * Events that will be executed after the broker has started. The are
     * usually set before the simulation start.
     */
    private List<PresetEvent> presetEvents = new ArrayList<>();

    /** If this broker has started receiving and responding to events. */
    private boolean started = false;

    /**
     * How lond we should keep this broker alive. If negative - the broker is
     * killed when no more cloudlets are left.
     */
    private final double lifeLength;

    /** Billing policy. */
    private IVmBillingPolicy vmBillingPolicy = null;

    /**
     * Constr.
     * 
     * @param name
     *            - the name of the broker.
     * @param lifeLength
     *            - for how long we need to keep this broker alive. If -1, then
     *            the broker is kept alive/running untill all cloudlets
     *            complete.
     * @throws Exception
     *             - from the superclass.
     */
    public DatacenterBrokerEX(final String name, final double lifeLength) throws Exception {
        super(name);
        this.lifeLength = lifeLength;
    }

    /**
     * Returns the billing policy, used by this broker.
     * 
     * @return the billing policy, used by this broker.
     */
    public IVmBillingPolicy getVMBillingPolicy() {
        return vmBillingPolicy;
    }

    /**
     * Sets the billing policy, used by this broker.
     * 
     * @param vmBillingPolicy
     *            - the new billing policy. If null - billing will be
     *            unavailable.
     */
    public void setVMBillingPolicy(IVmBillingPolicy vmBillingPolicy) {
        this.vmBillingPolicy = vmBillingPolicy;
    }

    /**
     * Returns the number of requested VM destructions.
     * 
     * @return the number of requested VM destructions.
     */
    public int getVmDestructsRequested() {
        return vmDestructsRequested;
    }

    /**
     * Sets the number of requested VM destructions.
     * 
     * @param vmDestructsRequested
     *            - the number of requested VM destructions. A valid positive
     *            integer or 0.
     */
    public void setVmDestructsRequested(int vmDestructsRequested) {
        this.vmDestructsRequested = vmDestructsRequested;
    }

    /**
     * Returns the number of acknowledged VM destructions.
     * 
     * @return the number of acknowledged VM destructions.
     */
    public int getVmDestructsAcks() {
        return vmDestructsAcks;
    }

    /**
     * Sets the number of acknowledged VM destructions.
     * 
     * @param vmDestructsAcks
     *            - acknowledged VM destructions. A valid positive integer or 0.
     */
    public void setVmDestructsAcks(int vmDestructsAcks) {
        this.vmDestructsAcks = vmDestructsAcks;
    }

    /**
     * Increments the counter of VM destruction acknowledgments.
     */
    protected void incrementVmDesctructsAcks() {
        vmDestructsAcks++;
    }

    /**
     * Returns if this broker has started to respond to events.
     * 
     * @return if this broker has started to respond to events.
     */
    protected boolean isStarted() {
        return started;
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (!started) {
            started = true;

            for (ListIterator<PresetEvent> iter = presetEvents.listIterator(); iter.hasNext();) {
                PresetEvent event = iter.next();
                send(event.id, event.delay, event.tag, event.data);
                iter.remove();
            }

            // Tell the broker to destroy itself after its lifeline.
            if (getLifeLength() > 0) {
                send(getId(), getLifeLength(), BROKER_DESTROY_ITSELF_NOW, null);
            }
        }

        switch (ev.getTag()) {
        case CloudSimTags.VM_CREATE_ACK:
            int[] data = (int[]) ev.getData();
            int vmId = data[1];

            Vm vm = VmList.getById(getVmList(), vmId);
            if (vm.isBeingInstantiated()) {
                vm.setBeingInstantiated(false);
            }
            processVmCreate(ev);
            break;

        default:
            super.processEvent(ev);
            break;
        }
    }

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        Cloudlet cloudlet = (Cloudlet) ev.getData();
        if (getLifeLength() <= 0) {
            // Will kill the broker if there are no more cloudlets.
            super.processCloudletReturn(ev);
        } else {

            getCloudletReceivedList().add(cloudlet);
            Log.printConcatLine(CloudSim.clock(), ": ", getName(), ": Cloudlet ", cloudlet.getCloudletId(), " received");
            cloudletsSubmitted--;

            // if (getCloudletList().size() > 0 && cloudletsSubmitted == 0) {
            // // all the cloudlets sent finished. It means that some bount
            // // cloudlet is waiting its VM be created
            // clearDatacenters();
            // createVmsInDatacenter(0);
            // }
        }
    }

    /**
     * Returns the list of preset events.
     * 
     * @return the list of preset events.
     */
    protected List<PresetEvent> getPresetEvents() {
        return presetEvents;
    }

    /**
     * Schedule an event that will be run with a given delay after the
     * simulation has started.
     * 
     * @param id
     * @param tag
     * @param data
     * @param delay
     */
    public void presetEvent(final int id, final int tag, final Object data, final double delay) {
        presetEvents.add(new PresetEvent(id, tag, data, delay));
    }

    /**
     * Submits the list of vms after a given delay
     * 
     * @param list
     * @param delay
     */
    public void createVmsAfter(List<? extends Vm> vms, double delay) {
        if (started) {
            send(getId(), delay, BROKER_SUBMIT_VMS_NOW, vms);
        } else {
            presetEvent(getId(), BROKER_SUBMIT_VMS_NOW, vms, delay);
        }
    }

    /**
     * Destroys the VMs after a specified time period. Used mostly for testing
     * purposes.
     * 
     * @param vms
     *            - the list of vms to terminate.
     * @param delay
     *            - the period to wait for.
     */
    public void destroyVMsAfter(final List<? extends Vm> vms, double delay) {
        if (started) {
            send(getId(), delay, BROKER_DESTROY_VMS_NOW, vms);
        } else {
            presetEvent(getId(), BROKER_DESTROY_VMS_NOW, vms, delay);
        }
    }

    /**
     * Submits the cloudlets after a specified time period. Used mostly for
     * testing purposes.
     * 
     * @param cloudlets
     *            - the cloudlets to submit.
     * @param delay
     *            - the delay.
     */
    public void submitCloudletList(List<Cloudlet> cloudlets, double delay) {
        if (started) {
            send(getId(), delay, BROKER_CLOUDLETS_NOW, cloudlets);
        } else {
            presetEvent(getId(), BROKER_CLOUDLETS_NOW, cloudlets, delay);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void processOtherEvent(SimEvent ev) {
        switch (ev.getTag()) {
        case CloudSimTags.VM_DESTROY_ACK:
            processVMDestroy(ev);
            break;
        case BROKER_DESTROY_VMS_NOW:
            destroyVMList((List<Vm>) ev.getData());
            break;
        case BROKER_SUBMIT_VMS_NOW:
            submitVmList((List<Vm>) ev.getData());
            // TODO Is the following valid when multiple data centres are
            // handled with a single broker?
            for (int nextDatacenterId : getDatacenterIdsList()) {
                createVmsInDatacenter(nextDatacenterId);
            }
            break;
        case BROKER_CLOUDLETS_NOW:
            submitCloudletList((List<Cloudlet>) ev.getData());
            submitCloudlets();
            break;
        case BROKER_DESTROY_ITSELF_NOW:
            closeDownBroker();
            break;
        default:
            super.processOtherEvent(ev);
            break;
        }
    }

    /**
     * Terminates the broker, releases all its resources and state.
     */
    public void closeDownBroker() {
        for (Vm vm : getVmList()) {
            finilizeVM(vm);
        }
        clearDatacenters();
        finishExecution();
    }

    private void processVMDestroy(SimEvent ev) {
        int[] data = (int[]) ev.getData();
        int datacenterId = data[0];
        int vmId = data[1];
        int result = data[2];

        if (result == CloudSimTags.TRUE) {
            Vm vm = VmList.getById(getVmsCreatedList(), vmId);

            // One more ack. to consider
            incrementVmDesctructsAcks();

            // Remove the vm from the created list
            getVmsCreatedList().remove(vm);
            finilizeVM(vm);

            // Kill all cloudlets associated with this VM
            for (Cloudlet cloudlet : getCloudletSubmittedList()) {
                if (!cloudlet.isFinished() && vmId == cloudlet.getVmId()) {
                    try {
                        vm.getCloudletScheduler().cloudletCancel(cloudlet.getCloudletId());
                        cloudlet.setCloudletStatus(Cloudlet.FAILED_RESOURCE_UNAVAILABLE);
                    } catch (Exception e) {
                        CustomLog.logError(Level.SEVERE, e.getMessage(), e);
                    }

                    sendNow(cloudlet.getUserId(), CloudSimTags.CLOUDLET_RETURN, cloudlet);
                }
            }

            // Use the standard log for consistency ....
            Log.printConcatLine(CloudSim.clock(), ": ", getName(), ": VM #", vmId,
                    " has been destroyed in Datacenter #", datacenterId);
        } else {
            // Use the standard log for consistency ....
            Log.printConcatLine(CloudSim.clock(), ": ", getName(), ": Desctuction of VM #", vmId,
                    " failed in Datacenter #", datacenterId);
        }

    }

    private void finilizeVM(final Vm vm) {
        if (vm instanceof VMex) {
            VMex vmEX = ((VMex) vm);
            if (vmEX.getStatus() != VMStatus.TERMINATED) {
                vmEX.setStatus(VMStatus.TERMINATED);
            }
        }
    }

    /**
     * Destroys/terminates the vms.
     * 
     * @param vms
     *            - the vms to terminate. Must not be null.
     */
    public void destroyVMList(final List<? extends Vm> vms) {
        if (getVmDestructsAcks() != getVmsDestroyed()) {
            throw new IllegalStateException("#" + getVmsDestroyed() + " have been marked for termination, but only #"
                    + getVmDestructsAcks() + " acknowlegdements have been received.");
        }

        int requestedVmTerminations = 0;
        for (final Vm vm : vms) {
            if (vm.getHost() == null || vm.getHost().getDatacenter() == null) {
                Log.print("VM " + vm.getId() + " has not been assigned in a valid way and can not be terminated.");
                continue;
            }

            // Update the cloudlets before we send the kill event
            vm.getHost().updateVmsProcessing(CloudSim.clock());

            int datacenterId = vm.getHost().getDatacenter().getId();
            String datacenterName = vm.getHost().getDatacenter().getName();

            Log.printConcatLine(CloudSim.clock(), ": ", getName(), ": Trying to Destroy VM #", vm.getId(), " in ",
                    datacenterName);

            // Tell the data centre to destroy it
            sendNow(datacenterId, CloudSimTags.VM_DESTROY_ACK, vm);
            requestedVmTerminations++;
        }

        setVmsDestroyed(requestedVmTerminations);
        setVmDestructsAcks(0);
    }

    public double getLifeLength() {
        return lifeLength;
    }

    /**
     * Bills the specified data centres with the specified policy. If no data
     * centres are specified - then all data centres billed.
     * <strong>NOTE:</strong> Before calling this method, you should set the
     * billing policy.
     * 
     * @param datacenterIds
     *            - the ids of the data centres.
     * @return the incurred debt.
     * @throws NullPointerException
     *             - if the billing policy has not been set.
     */
    public BigDecimal bill(final Integer... datacenterIds) {
        Set<Integer> dcIds = new HashSet<>(Arrays.asList(datacenterIds));
        List<Vm> toBill = new ArrayList<>();

        for (Vm vm : getVmList()) {
            if (dcIds.isEmpty() || dcIds.contains(getVmsToDatacentersMap().get(vm.getId()))) {
                toBill.add(vm);
            }
        }

        return vmBillingPolicy.bill(toBill);
    }

    @Override
    public String toString() {
        return String.valueOf(String.format("Broker(%s, %d)", Objects.toString(getName(), "N/A"), getId()));
    }

    /**
     * CloudSim does not execute events that are fired before the simulation has
     * started. Thus we need to buffer them and then refire when the simulation
     * starts.
     * 
     * @author nikolay.grozev
     * 
     */
    protected static class PresetEvent {
        final int id;
        final int tag;
        final Object data;
        final double delay;

        public PresetEvent(final int id, final int tag, final Object data, final double delay) {
            super();
            this.id = id;
            this.tag = tag;
            this.data = data;
            this.delay = delay;
        }
    }
}
