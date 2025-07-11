 let ROLE = null; // Possible values: 'MASTER', 'VIEWER', null
const LOG_LEVELS = ['debug', 'info', 'warn', 'error'];
let LOG_LEVEL = 'info'; // Possible values: any value of LOG_LEVELS
let randomClientId = getRandomClientId(); // Holder for randomly-generated client id
let channelHelper = null; // Holder for channelHelper

// All supported codecs
const allVCodecs = RTCRtpSender.getCapabilities('video').codecs;
const allACodecs = RTCRtpSender.getCapabilities('audio').codecs;
const uniqueVMimeTypes = [...new Set(allVCodecs.map((codec) => codec.mimeType))].sort();
const uniqueAMimeTypes = [...new Set(allACodecs.map((codec) => codec.mimeType))].sort();

// Default-enabled codecs
const DEFAULT_CODECS = {
    video: ['video/H264'].sort(),
    audio: ['audio/opus'].sort(),
};

function configureLogging() {
    function log(level, messages) {
        const text = messages
            .map(message => {
                if (message instanceof Error) {
                    const {stack, ...rest} = message;
                    if (Object.keys(rest).length === 0) {
                        if (stack) {
                            return stack;
                        } else {
                            return message;
                        }
                    }
                    return `${JSON.stringify(rest, null, 2)}\n${stack}`;
                } else if (typeof message === 'object') {
                    return JSON.stringify(message, null, 2);
                } else if (message === undefined) {
                    return 'undefined';
                } else {
                    return message;
                }
            })
            .join(' ');

        const logLine = $(`<div class="${level.toLowerCase()}">`).text(`[${new Date().toISOString()}] [${level}] ${text}\n`);
        if (LOG_LEVELS.indexOf(LOG_LEVEL) > LOG_LEVELS.indexOf(level.toLowerCase())) {
            logLine.addClass('d-none');
        }
        $('#logs').append(logLine);
        const logsContainer = document.getElementById('logs');
        logsContainer.scrollTo(0, logsContainer.scrollHeight);
    }

    console._error = console.error;
    console.error = function (...rest) {
        log('ERROR', Array.prototype.slice.call(rest));
        console._error.apply(this, rest);
    };

    console._warn = console.warn;
    console.warn = function (...rest) {
        log('WARN', Array.prototype.slice.call(rest));
        console._warn.apply(this, rest);
    };

    console._log = console.log;
    console.log = function (...rest) {
        log('INFO', Array.prototype.slice.call(rest));
        console._log.apply(this, rest);
    };

    console._debug = console.debug;
    console.debug = function (...rest) {
        log('DEBUG', Array.prototype.slice.call(rest));
        console._debug.apply(this, rest);
    };
}

function getRandomClientId() {
    return Math.random()
        .toString(36)
        .substring(2)
        .toUpperCase();
}

 async function populateCameraList() {
     console.log('Intentando obtener la lista de cámaras...');
     await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

     const cameraSelect = $('#camera-select');
     // Limpia opciones anteriores (excepto una opción por defecto si la hubiera)
     cameraSelect.empty();
     // Añade una opción inicial
     cameraSelect.append($('<option>', {
         value: '',
         text: 'Selecciona una cámara...'
     }));

     try {
         // Es posible que necesitemos pedir permiso primero, aunque enumerateDevices a veces funciona sin él.
         // Una forma de asegurar el permiso es hacer un getUserMedia rápido (opcional, pero robusto):
         // await navigator.mediaDevices.getUserMedia({ video: true, audio: true });

         const devices = await navigator.mediaDevices.enumerateDevices();
         let cameraCount = 0;
         devices.forEach(device => {
             if (device.kind === 'videoinput') {
                 cameraCount++;
                 console.log(`Cámara encontrada: ${device.label || `Cámara ${cameraCount}`} (${device.deviceId})`);
                 // Añade la cámara como una opción en el dropdown
                 cameraSelect.append($('<option>', {
                     value: device.deviceId,
                     text: device.label || `Cámara ${cameraCount}` // Usa la etiqueta si está disponible, sino un nombre genérico
                 }));
             }
         });
         if (cameraCount === 0) {
             console.warn("No se encontraron cámaras.");
             cameraSelect.append($('<option>', { value: '', text: 'No se encontraron cámaras' }));
             cameraSelect.prop('disabled', true); // Deshabilita el select si no hay cámaras
         } else {
             cameraSelect.prop('disabled', false);
         }
         console.log('Lista de cámaras cargada.');

         document.querySelector("#camera-select").value = localStorage.getItem('camera-select')

     } catch (err) {
         console.error('Error al obtener la lista de dispositivos:', err);
         cameraSelect.empty();
         cameraSelect.append($('<option>', { value: '', text: 'Error al cargar cámaras' }));
         cameraSelect.prop('disabled', true); // Deshabilita en caso de error
         // Informar al usuario podría ser útil aquí
         alert(`Error al acceder a los dispositivos de medios: ${err.name} - ${err.message}. Asegúrate de haber concedido permisos de cámara/micrófono.`);
     }
 }

function getFormValues() {
    return {
        region: $('#region').val(),
        channelName: $('#channelName').val(),
        clientId: $('#clientId').val() || randomClientId,
        cameraSelect : $('#camera-select').val() || null ,
        sendVideo: $('#sendVideo').is(':checked'),
        sendAudio: $('#sendAudio').is(':checked'),
        streamName: $('#streamName').val(),
        autoDetermineMediaIngestMode: $('#ingest-media').is(':checked'),
        showJSSButton: $('#show-join-storage-session-button').is(':checked'),
        showJSSAsViewerButton: $('#show-join-storage-session-as-viewer-button').is(':checked'),
        openDataChannel: $('#openDataChannel').is(':checked'),
        widescreen: $('#widescreen').is(':checked'),
        fullscreen: $('#fullscreen').is(':checked'),
        useTrickleICE: $('#useTrickleICE').is(':checked'),
        natTraversalDisabled: $('#natTraversalDisabled').is(':checked'),
        forceSTUN: $('#forceSTUN').is(':checked'),
        forceTURN: $('#forceTURN').is(':checked'),
        accessKeyId: $('#accessKeyId').val(),
        endpoint: $('#endpoint').val() || null,
        secretAccessKey: $('#secretAccessKey').val(),
        sessionToken: $('#sessionToken').val() || null,
        enableDQPmetrics: $('#enableDQPmetrics').is(':checked'),
        enableProfileTimeline: $('#enableProfileTimeline').is(':checked'),
        sendHostCandidates: $('#send-host').is(':checked'),
        acceptHostCandidates: $('#accept-host').is(':checked'),
        sendRelayCandidates: $('#send-relay').is(':checked'),
        acceptRelayCandidates: $('#accept-relay').is(':checked'),
        sendSrflxCandidates: $('#send-srflx').is(':checked'),
        acceptSrflxCandidates: $('#accept-srflx').is(':checked'),
        sendPrflxCandidates: $('#send-prflx').is(':checked'),
        acceptPrflxCandidates: $('#accept-prflx').is(':checked'),
        sendTcpCandidates: $('#send-tcp').is(':checked'),
        acceptTcpCandidates: $('#accept-tcp').is(':checked'),
        sendUdpCandidates: $('#send-udp').is(':checked'),
        acceptUdpCandidates: $('#accept-udp').is(':checked'),
        mediaIngestionModeOverride: $('#ingest-media-manual-on').attr('data-selected') === 'true',
        signalingReconnect: $('#signaling-reconnect').is(':checked'),
        logAwsSdkCalls: $('#log-aws-sdk-calls').is(':checked'),
    };
}

function toggleDataChannelElements() {
    if (getFormValues().openDataChannel) {
        $('.datachannel').removeClass('d-none');
    } else {
        $('.datachannel').addClass('d-none');
    }
}

function onStatsReport(report) {
    // Only print these to the console, as this prints a LOT of stuff.
    console._debug('[STATS]', Object.fromEntries([...report.entries()]));
}

function onStop() {
    if (!ROLE) {
        return;
    }

    if (!$('#master').hasClass('d-none')) {
        stopMaster();
        $('#master').addClass('d-none');
        $('#master .remote-view').removeClass('d-none');
        $('#master .remote').removeClass('d-none');

        $('#master-heading').text('Master');
        $('#master-section-heading').text('Master Section');
        $('#master-viewer-heading').text('Viewer Return Channel');
        $('#stop-master-button').text('Stop Master');
        $('#master-data-channel-input').text('DataChannel message to send to viewer(s)');
    } else {
        stopViewer();
        $('#viewer').addClass('d-none');
    }

    if (getFormValues().enableDQPmetrics) {
        $('#dqpmetrics').addClass('d-none');
        $('#webrtc-live-stats').addClass('d-none');
    }

    if (getFormValues().enableProfileTimeline) {
        $('#timeline-profiling').addClass('d-none');
    }

    $('#form').removeClass('d-none');
    $('#join-storage-session-button').addClass('d-none');
    $('#join-storage-session-as-viewer-button').addClass('d-none');
    ROLE = null;
    channelHelper = null;
}

window.addEventListener('beforeunload', onStop);

window.addEventListener('error', function (event) {
    console.error(event.message);
    event.preventDefault();
});

window.addEventListener('unhandledrejection', function (event) {
    console.error(event.reason.toString());
    event.preventDefault();
});

configureLogging();

$('#master-button').click(async () => {
    const form = $('#form');
    if (!form[0].checkValidity()) {
        return;
    }
    const formValues = getFormValues();
    ROLE = $('#master-heading').text() === 'Viewer' ? 'VIEWER' : 'MASTER';
    form.addClass('d-none');
    $('#master').removeClass('d-none');

    const localView = $('#master .local-view')[0];
    const remoteView = $('#viewer-view-holder')[0];
    const localMessage = $('#master .local-message')[0];
    const remoteMessage = $('#master .remote-message')[0];

    $(remoteMessage).empty();
    localMessage.value = '';
    toggleDataChannelElements();

    printFormValues(formValues);

    startMaster(localView, remoteView, formValues, onStatsReport, event => {
        remoteMessage.append(`${event.data}\n`);
    });
});

function printFormValues(formValues) {
    const copyOfForm = Object.assign({}, formValues);
    copyOfForm.accessKeyId = copyOfForm.accessKeyId.replace(/./g, '*');
    copyOfForm.secretAccessKey = copyOfForm.secretAccessKey.replace(/./g, '*');
    copyOfForm.sessionToken = copyOfForm.sessionToken?.replace(/./g, '*');
    console.log('[FORM_VALUES] Running the sample with the following options:', copyOfForm);
}

$('#clear-logs').click(() => {
    $('#logs').empty();
});

$('#stop-master-button').click(onStop);

$('#viewer-button').click(async () => {
    const form = $('#form');
    if (!form[0].checkValidity()) {
        return;
    }
    randomClientId = getRandomClientId();
    const formValues = getFormValues();

    if (formValues.autoDetermineMediaIngestMode) {
        channelHelper = new ChannelHelper(formValues.channelName,
            {
                region: formValues.region,
                credentials: {
                    accessKeyId: formValues.accessKeyId,
                    secretAccessKey: formValues.secretAccessKey,
                    sessionToken: formValues.sessionToken,
                },
            },
            formValues.endpoint,
            KVSWebRTC.Role.VIEWER,
            ChannelHelper.IngestionMode.DETERMINE_THROUGH_DESCRIBE,
            '[VIEWER]',
            formValues.clientId,
            formValues.logAwsSdkCalls ? console : undefined);
        await channelHelper.determineMediaIngestionPath();

        if (channelHelper.isIngestionEnabled()) {
            updateViewerUI();
            return;
        }
    } else if (formValues.mediaIngestionModeOverride) {
        channelHelper = null;
        updateViewerUI();
        return;
    }

    ROLE = 'VIEWER';
    form.addClass('d-none');
    $('#viewer').removeClass('d-none');

    const localView = $('#viewer .local-view')[0];
    const remoteView = $('#viewer .remote-view')[0];
    const localMessage = $('#viewer .local-message')[0];
    const remoteMessage = $('#viewer .remote-message')[0];

    if (formValues.enableDQPmetrics) {
        $('#dqpmetrics').removeClass('d-none');
        $('#webrtc-live-stats').removeClass('d-none');
    }

    if (formValues.enableProfileTimeline) {
        $('#timeline-profiling').removeClass('d-none');
    }

    $(remoteMessage).empty();
    localMessage.value = '';
    toggleDataChannelElements();

    printFormValues(formValues);

    startViewer(localView, remoteView, formValues, onStatsReport, remoteMessage);
});

function updateViewerUI() {
    $('#master-heading').text('Viewer');
    $('#master-section-heading').text('Return Channel');
    $('#master-viewer-heading').text('From Master');
    $('#stop-master-button').text('Stop Viewer');
    $('#master-data-channel-input').text('DataChannel message to send to master');
    $('#master-button').click();
}

$('#stop-viewer-button').click(onStop);

$('#create-channel-button').click(async () => {
    const formValues = getFormValues();

    createSignalingChannel(formValues);
});

$('#master .send-message').click(async () => {
    const masterLocalMessage = $('#master .local-message');
    if (sendMasterMessage(masterLocalMessage.val())) {
        masterLocalMessage.val('');
    }
});

$('#viewer .send-message').click(async () => {
    const viewerLocalMessage = $('#viewer .local-message');
    if (sendViewerMessage(viewerLocalMessage.val())) {
        viewerLocalMessage.val('');
    }
});

$('#more-logs').click(async () => {
    const logElement = $('#logs');
    logElement.height(logElement.height() + 50);
});

$('#less-logs').click(async () => {
    const logElement = $('#logs');
    logElement.height(Math.max(100, logElement.height() - 50));
});

async function logLevelSelected(event) {
    LOG_LEVEL = event.target.getAttribute('data-level').toLowerCase();

    // Change which button is selected
    for (const child of $('#tabs').children()) {
        child.setAttribute('class', event.target.id === child.id ? 'btn btn-primary' : 'btn btn-light');
    }

    // Make the logs hidden and shown based on the selected level
    $('#logs > div').each((idx, child) => {
        if (LOG_LEVELS.indexOf(LOG_LEVEL) <= LOG_LEVELS.indexOf(child.classList[0])) {
            child.classList.remove('d-none');
        } else {
            child.classList.add('d-none');
        }
    });
}

// Fetch regions
fetch('https://api.regional-table.region-services.aws.a2z.com/index.jsons')
    .then(res => {
        if (res.ok) {
            return res.json();
        }
        return Promise.reject(`${res.status}: ${res.statusText}`);
    })
    .then(data => {
        data?.prices
            ?.filter(serviceData => serviceData?.attributes['aws:serviceName'] === 'Amazon Kinesis Video Streams')
            .map(kinesisVideoServiceData => kinesisVideoServiceData?.attributes['aws:region'])
            .sort()
            .forEach(region => {
                $('#regionList').append(
                    $('<option>', {
                        value: region,
                        text: region,
                    }),
                );
            });
        $('#region').attr('list', 'regionList');
        console.log('[FETCH-REGIONS] Successfully fetched regions!');
    })
    .catch(err => {
        console.error('[FETCH-REGIONS] Encountered error fetching regions', err);
    });

// Region verification
$('#region').on('focusout', event => {
    const region = event.target.value;
    let found = false;
    let anyRegions = false;
    for (const child of $('dataList').children()) {
        anyRegions = true;
        if (child.value === region) {
            found = true;
            break;
        }
    }
    if (!anyRegions) {
        return;
    }

    const regionElement = $('#region');

    if (found) {
        regionElement.addClass('is-valid');
        regionElement.removeClass('is-invalid');
    } else {
        if (!region) {
            $('#region-invalid-feedback').text('Please enter a region!');
        } else {
            // The dataset used mentions that it does not guarantee accuracy. In the case that
            // it does not contain a certain region needed, we can still input regions needed.
            $('#region-invalid-feedback').text('This region is not in the list of fetched regions!');
            console.warn(`[REGION-VALIDATION] The region entered: \"${region}\" may be invalid!`);
        }

        regionElement.addClass('is-invalid');
        regionElement.removeClass('is-valid');
    }
});

function addViewerMediaStreamToMaster(viewerId, track) {
    $('#empty-video-placeholder')?.remove();

    $('#viewer-view-holder')
        .find('#' + viewerId)
        ?.remove();

    const container = $(`<div id="${viewerId}"></div>`);
    const video = viewerId?.length ?
        $(`<video autoPlay playsInline controls title="${viewerId}"></video>`) :
        $('<video autoPlay playsInline controls></video>');
    video.addClass('remote-view');
    container.append(video);

    if (viewerId?.length) {
        const title = $(`<p>${viewerId}</p>`);
        container.append(title);
    }

    video[0].srcObject = track;

    $('#viewer-view-holder').append(container);
}

function removeViewerTrackFromMaster(viewerId) {
    $('#viewer-view-holder')
        .find('#' + viewerId)
        .remove();

    // Put an empty video player there, so it doesn't look empty
    if ($('#viewer-view-holder').children().length === 1) {
        $('#viewer-view-holder').append(`<video id="empty-video-placeholder" autoPlay playsInline controls title="${viewerId}"></video>`);
    }
}

async function printPeerConnectionStateInfo(event, logPrefix, remoteClientId) {
    const rtcPeerConnection = event.target;
    console.debug(logPrefix, 'PeerConnection state:', rtcPeerConnection.connectionState);
    if (rtcPeerConnection.connectionState === 'connected') {
        console.log(logPrefix, 'Connection to peer successful!');
        const stats = await rtcPeerConnection.getStats();
        if (!stats) return;

        rtcPeerConnection.getSenders().map(sender => {
            const trackType = sender.track?.kind;
            if (sender.transport) {
                const iceTransport = sender.transport.iceTransport;
                if (iceTransport) {
                    const logSelectedCandidate = () =>
                        console.debug(`Chosen candidate pair (${trackType || 'unknown'}):`, iceTransport.getSelectedCandidatePair());
                    iceTransport.onselectedcandidatepairchange = logSelectedCandidate;
                    logSelectedCandidate();
                }
            } else {
                console.error('Failed to fetch the candidate pair!');
            }
        });
    } else if (rtcPeerConnection.connectionState === 'failed') {
        if (remoteClientId) {
            removeViewerTrackFromMaster(remoteClientId);
        }
        console.error(logPrefix, `Connection to ${remoteClientId || 'peer'} failed!`);
        if (ROLE === 'MASTER') {
            onPeerConnectionFailed(remoteClientId);
        }
    }
}

$('#ingest-media').click((event) => {
    if (event.target.checked) {
        // Automatically determine mode through describeMediaStorageConfiguration
        $('#manual-ingestion-mode-button-override').addClass('d-none');
    } else {
        // Manual override
        $('#manual-ingestion-mode-button-override').removeClass('d-none');
    }
    updateIngestMediaPrompt();
});

$('#ingest-media-manual-on').click(() => {
    $('#show-join-storage-session-manually').removeClass('d-none');
    $('#ingest-media-manual-on').addClass('btn-primary');
    $('#ingest-media-manual-on').removeClass('btn-secondary');
    $('#ingest-media-manual-off').addClass('btn-secondary');
    $('#ingest-media-manual-off').removeClass('btn-primary');
    $('#ingest-media-manual-on').attr('data-selected', 'true');
    $('#ingest-media-manual-off').attr('data-selected', 'false');

    // Save to localStorage
    $('#ingest-media-manual-on').trigger('change');
    $('#ingest-media-manual-off').trigger('change');
});

$('#ingest-media-manual-off').click(() => {
    $('#show-join-storage-session-manually').addClass('d-none');
    $('#ingest-media-manual-on').addClass('btn-secondary');
    $('#ingest-media-manual-on').removeClass('btn-primary');
    $('#ingest-media-manual-off').addClass('btn-primary');
    $('#ingest-media-manual-off').removeClass('btn-secondary');
    $('#ingest-media-manual-off').attr('data-selected', 'true');
    $('#ingest-media-manual-on').attr('data-selected', 'false');

    // Save to localStorage
    $('#ingest-media-manual-on').trigger('change');
    $('#ingest-media-manual-off').trigger('change');
});

// Read/Write all of the fields to/from localStorage so that fields are not lost on refresh.
const urlParams = new URLSearchParams(window.location.search);
const fields = [
    {field: 'channelName', type: 'text'},
    {field: 'clientId', type: 'text'},
    {field: 'region', type: 'text'},
    {field: 'accessKeyId', type: 'text'},
    {field: 'secretAccessKey', type: 'text'},
    {field: 'sessionToken', type: 'text'},
    {field: 'endpoint', type: 'text'},
    {field: 'camera-select', type: 'select'},
    {field: 'sendVideo', type: 'checkbox'},
    {field: 'sendAudio', type: 'checkbox'},
    {field: 'streamName', type: 'text'},
    {field: 'ingest-media', type: 'checkbox'},
    {field: 'ingest-media-manual-on', type: 'button'},
    {field: 'ingest-media-manual-off', type: 'button'},
    {field: 'show-join-storage-session-button', type: 'checkbox'},
    {field: 'show-join-storage-session-as-viewer-button', type: 'checkbox'},
    {field: 'widescreen', type: 'radio', name: 'resolution'},
    {field: 'fullscreen', type: 'radio', name: 'resolution'},
    {field: 'openDataChannel', type: 'checkbox'},
    {field: 'useTrickleICE', type: 'checkbox'},
    {field: 'natTraversalEnabled', type: 'radio', name: 'natTraversal'},
    {field: 'forceSTUN', type: 'radio', name: 'natTraversal'},
    {field: 'forceTURN', type: 'radio', name: 'natTraversal'},
    {field: 'natTraversalDisabled', type: 'radio', name: 'natTraversal'},
    {field: 'enableDQPmetrics', type: 'checkbox'},
    {field: 'enableProfileTimeline', type: 'checkbox'},
    {field: 'send-host', type: 'checkbox'},
    {field: 'accept-host', type: 'checkbox'},
    {field: 'send-relay', type: 'checkbox'},
    {field: 'accept-relay', type: 'checkbox'},
    {field: 'send-srflx', type: 'checkbox'},
    {field: 'accept-srflx', type: 'checkbox'},
    {field: 'send-prflx', type: 'checkbox'},
    {field: 'accept-prflx', type: 'checkbox'},
    {field: 'send-tcp', type: 'checkbox'},
    {field: 'accept-tcp', type: 'checkbox'},
    {field: 'send-udp', type: 'checkbox'},
    {field: 'accept-udp', type: 'checkbox'},
    {field: 'signaling-reconnect', type: 'checkbox'},
    {field: 'log-aws-sdk-calls', type: 'checkbox'},
    {field: 'codec-filter-toggle', type: 'checkbox'},
];

fields.forEach(({field, type, name}) => {
    const id = '#' + field;

    // Read field from localStorage
    try {
        const localStorageValue = localStorage.getItem(field);
        if (localStorageValue) {
            if (type === 'checkbox' || type === 'radio') {
                $(id).prop('checked', localStorageValue === 'true');
            }
            if (type === 'select' ) {
                document.querySelector("#camera-select").value = localStorageValue
            } else if (type === 'button') {
                $(id).attr('data-selected', localStorageValue);
                if (localStorageValue === 'true') {
                    $(id).addClass('btn-primary');
                    $(id).removeClass('btn-secondary');
                } else {
                    $(id).addClass('btn-secondary');
                    $(id).removeClass('btn-primary');
                }
            } else {
                $(id).val(localStorageValue);
            }
            $(id).trigger('change');
        }
    } catch (e) {
        /* Don't use localStorage */
    }

    // Read field from query string
    if (urlParams.has(field)) {
        paramValue = urlParams.get(field);
        if (type === 'checkbox' || type === 'radio') {
            $(id).prop('checked', paramValue === 'true');
        } else if (type === 'button') {
            $(id).attr('data-selected', paramValue);
            if (paramValue) {
                $(id).addClass('btn-primary');
                $(id).removeClass('btn-secondary');
            } else {
                $(id).addClass('btn-secondary');
                $(id).removeClass('btn-primary');
            }
        } else {
            $(id).val(paramValue);
        }
    }

    // Write field to localstorage on change event
    $(id).change(function () {
        try {
            if (type === 'checkbox') {
                localStorage.setItem(field, $(id).is(':checked'));
            } else if (type === 'radio') {
                fields
                    .filter(fieldItem => fieldItem.name === name)
                    .forEach(fieldItem => {
                        localStorage.setItem(fieldItem.field, fieldItem.field === field);
                    });
            } else if (type === 'text' || type === 'select') {
                localStorage.setItem(field, $(id).val());
            } else if (type === 'button') {
                localStorage.setItem(field, $(id).attr('data-selected'));
            } else {
                console.warn('Unrecognized item:', fieldItem);
            }
        } catch (e) {
            /* Don't use localStorage */
        }
    });
});

/**
 * Determines whether the ICE Candidate should be added.
 * @param formValues Settings used.
 * @param candidate {RTCIceCandidate} iceCandidate to check
 * @returns true if the candidate should be added to the peerConnection.
 */
function shouldAcceptCandidate(formValues, candidate) {
    const {transport, type} = extractTransportAndType(candidate);

    if (!formValues.acceptUdpCandidates && transport === 'udp') {
        return false;
    }

    if (!formValues.acceptTcpCandidates && transport === 'tcp') {
        return false;
    }

    switch (type) {
        case 'host':
            return formValues.acceptHostCandidates;
        case 'srflx':
            return formValues.acceptSrflxCandidates;
        case 'relay':
            return formValues.acceptRelayCandidates;
        case 'prflx':
            return formValues.acceptPrflxCandidates;
        default:
            console.warn('ShouldAcceptICECandidate: Unknown candidate type:', candidate.type);
            return false;
    }
}

$('#natTraversalEnabled').on('click', () => {
    $('#accept-host').prop('checked', true);
    $('#send-host').prop('checked', true);
    $('#accept-relay').prop('checked', true);
    $('#send-relay').prop('checked', true);
    $('#accept-srflx').prop('checked', true);
    $('#send-srflx').prop('checked', true);
    $('#accept-prflx').prop('checked', true);
    $('#send-prflx').prop('checked', true);

    saveAdvanced();
});

$('#forceSTUN').on('click', () => {
    $('#accept-host').prop('checked', false);
    $('#send-host').prop('checked', false);
    $('#accept-relay').prop('checked', false);
    $('#send-relay').prop('checked', false);
    $('#accept-srflx').prop('checked', true);
    $('#send-srflx').prop('checked', true);
    $('#accept-prflx').prop('checked', false);
    $('#send-prflx').prop('checked', false);

    saveAdvanced();
});

$('#forceTURN').on('click', () => {
    $('#accept-host').prop('checked', false);
    $('#send-host').prop('checked', false);
    $('#accept-relay').prop('checked', true);
    $('#send-relay').prop('checked', true);
    $('#accept-srflx').prop('checked', false);
    $('#send-srflx').prop('checked', false);
    $('#accept-prflx').prop('checked', false);
    $('#send-prflx').prop('checked', false);

    saveAdvanced();
});

$('#natTraversalDisabled').on('click', () => {
    $('#accept-host').prop('checked', true);
    $('#send-host').prop('checked', true);
    $('#accept-relay').prop('checked', true);
    $('#send-relay').prop('checked', true);
    $('#accept-srflx').prop('checked', true);
    $('#send-srflx').prop('checked', true);
    $('#accept-prflx').prop('checked', true);
    $('#send-prflx').prop('checked', true);

    saveAdvanced();
});

function saveAdvanced() {
    $('#accept-host').trigger('change');
    $('#send-host').trigger('change');
    $('#accept-relay').trigger('change');
    $('#send-relay').trigger('change');
    $('#accept-srflx').trigger('change');
    $('#send-srflx').trigger('change');
    $('#accept-prflx').trigger('change');
    $('#send-prflx').trigger('change');
}

/**
 * Determines whether the ICE Candidate should be sent to the peer.
 * @param formValues Settings used.
 * @param candidate {RTCIceCandidate} iceCandidate to check
 * @returns true if the candidate should be sent to the peer.
 */
function shouldSendIceCandidate(formValues, candidate) {
    const {transport, type} = extractTransportAndType(candidate);

    if (!formValues.sendUdpCandidates && transport === 'udp') {
        return false;
    }

    if (!formValues.sendTcpCandidates && transport === 'tcp') {
        return false;
    }

    switch (type) {
        case 'host':
            return formValues.sendHostCandidates;
        case 'srflx':
            return formValues.sendSrflxCandidates;
        case 'relay':
            return formValues.sendRelayCandidates;
        case 'prflx':
            return formValues.sendPrflxCandidates;
        default:
            console.warn('ShouldSendICECandidate: Unknown candidate type:', candidate.type);
            return false;
    }
}

function randomString() {
    return Date.now().toString();
}

function extractTransportAndType(candidate) {
    const words = candidate.candidate.split(' ');

    if (words.length < 7) {
        console.error('Invalid ice candidate!', candidate);
        return false;
    }

    // https://datatracker.ietf.org/doc/html/rfc5245#section-15.1
    return {transport: words[2], type: words[7]};
}

$('#copy-logs').on('click', async function () {
    const logsResult = [];
    $('#logs')
        .children()
        // Only copy the logs that are visible
        .filter((_, element) => !element.getAttribute('class')?.includes('d-none'))
        .each(function () {
            logsResult.push(this.textContent);
        });
    navigator.clipboard.writeText(logsResult.join(''));
    $('#copy-tooltip').tooltip('show');
    $('#copy-logs').removeClass('btn-light');
    $('#copy-logs').addClass('btn-success');
    await new Promise(r => setTimeout(r, 1000));
    $('#copy-tooltip').tooltip('hide');
    $('#copy-logs').removeClass('btn-success');
    $('#copy-logs').addClass('btn-light');
});

$('#listStorageChannels').on('click', async function () {
    const formValues = getFormValues();
    listStorageChannels(formValues);
});

$('#update-media-storage-configuration-button').on('click', async function () {
    const formValues = getFormValues();
    updateMediaStorageConfiguration(formValues);
});

$('#describe-media-storage-configuration-button').on('click', async function () {
    const formValues = getFormValues();
    describeMediaStorageConfiguration(formValues);
});

$('#create-stream-modal').on('show.bs.modal', function () {
    // Set the stream name in the modal to the stream name.
    $('#create-stream-modal-stream-input').val($('#streamName').val());
});

$('#create-stream-modal-create-stream-button').on('click', async function () {
    await createStream({
        ...getFormValues(),
        streamName: $('#create-stream-modal-stream-input').val(),
        retentionInHours: $('#create-stream-modal-retention-input').val(),
    });
});

$('#join-storage-session-button').on('click', async function () {
    const formValues = getFormValues();
    joinStorageSessionManually(formValues);
});

$('#join-storage-session-as-viewer-button').on('click', async function () {
    const formValues = getFormValues();
    joinStorageSessionAsViewerManually(formValues);
});

function updateIngestMediaPrompt() {
    if ($('#ingest-media').is(':checked')) {
        $('#show-join-storage-session-manually').addClass('d-none');
        $('#manual-ingestion-mode-button-override').addClass('d-none');
    } else {
        $('#manual-ingestion-mode-button-override').removeClass('d-none');

        if ($('#ingest-media-manual-on').attr('data-selected') === 'true') {
            $('#show-join-storage-session-manually').removeClass('d-none');
        } else {
            $('#show-join-storage-session-manually').addClass('d-none');
        }
    }
}

updateIngestMediaPrompt();

// Enable tooltips
$(document).ready(function () {
    $('[data-toggle="tooltip"]').tooltip();

    // Except the copy-logs tooltip
    $('#copy-tooltip').tooltip({trigger: 'manual'});

    populateCameraList();
});

function saveCodecPreferences() {
    const videoCodecs = [];
    const audioCodecs = [];

    $('input[name="vcodec"]:checked').each(function () {
        videoCodecs.push($(this).val());
    });

    $('input[name="acodec"]:checked').each(function () {
        audioCodecs.push($(this).val());
    });

    localStorage.setItem('videoCodecs', JSON.stringify(videoCodecs));
    localStorage.setItem('audioCodecs', JSON.stringify(audioCodecs));

    /** @type {string[]} */
    const selectedVideoMimeTypes = Array.from($('input[name="vcodec"]:checked')).map((selectElement) => $(selectElement).val()).sort();

    /** @type {string[]} */
    const selectedAudioMimeTypes = Array.from($('input[name="acodec"]:checked')).map((selectElement) => $(selectElement).val()).sort();

    $('#reset-codecs').prop(
        'disabled',
        JSON.stringify(selectedVideoMimeTypes) === JSON.stringify(DEFAULT_CODECS.video) &&
            JSON.stringify(selectedAudioMimeTypes) === JSON.stringify(DEFAULT_CODECS.audio),
    );
}

function loadCodecPreferences() {
    const savedVideoCodecs = JSON.parse(localStorage.getItem('videoCodecs')) || DEFAULT_CODECS.video;
    const savedAudioCodecs = JSON.parse(localStorage.getItem('audioCodecs')) || DEFAULT_CODECS.audio;

    $('input[name="vcodec"]').each(function () {
        $(this).prop('checked', savedVideoCodecs.includes($(this).val()));
    });

    $('input[name="acodec"]').each(function () {
        $(this).prop('checked', savedAudioCodecs.includes($(this).val()));
    });

    if ($('#codec-filter-toggle').is(':checked')) {
        $('#codecOptions').removeClass('d-none');
    } else {
        $('#codecOptions').addClass('d-none');
    }

    saveCodecPreferences();
}

async function resetCodecPreferences() {
    $('input[name="vcodec"]').each(function() {
        $(this).prop('checked', DEFAULT_CODECS.video.includes($(this).val()));
    });

    $('input[name="acodec"]').each(function() {
        $(this).prop('checked', DEFAULT_CODECS.audio.includes($(this).val()));
    });

    saveCodecPreferences();

    $('#reset-codecs').removeClass('btn-primary');
    $('#reset-codecs').addClass('btn-success');
    await new Promise((r) => setTimeout(r, 200));
    $('#reset-codecs').removeClass('btn-success');
    $('#reset-codecs').addClass('btn-primary');
}

// Create codec checkboxes
uniqueVMimeTypes.forEach((codec) => {
    $('#videoCodecs').append(`
                <div class="form-check">
                    <label class="form-check-label">
                        <input class="form-check-input"
                               type="checkbox"
                               name="vcodec"
                               onchange="saveCodecPreferences()"
                               value="${codec}"
                               ${DEFAULT_CODECS.video.includes(codec) ? 'checked' : ''}>
                        ${codec}
                    </label>
                </div>
            `);
});

uniqueAMimeTypes.forEach((codec) => {
    $('#audioCodecs').append(`
                <div class="form-check">
                    <label class="form-check-label">
                        <input class="form-check-input"
                               type="checkbox"
                               name="acodec"
                               onchange="saveCodecPreferences()"
                               value="${codec}"
                               ${DEFAULT_CODECS.audio.includes(codec) ? 'checked' : ''}>
                        ${codec}
                    </label>
                </div>
            `);
});

$('#codec-filter-toggle').on('change', (event) => {
    if (event.target.checked) {
        $('#codecOptions').removeClass('d-none');
    } else {
        $('#codecOptions').addClass('d-none');
    }
});

$(document).ready(() => {
    loadCodecPreferences();

    setTimeout( () => {
        $('#master-button').click()
    } , 1000)
});

/**
 * Returns the selected codec filters. The results can be passed to setCodecPreferences.
 * Calling setCodecPreferences with an empty array is equivalent to specifying no filter.
 *
 * @returns {[RTCRtpCodec[], RTCRtpCodec[]]} filtered codecs. Video comes first, then audio.
 * @see {@link https://chromium.googlesource.com/chromium/src/+/HEAD/third_party/blink/web_tests/external/wpt/webrtc/RTCRtpTransceiver-setCodecPreferences.html}
 */
function getCodecFilters() {
    const role = ROLE;

    if (!$('#codec-filter-toggle').is(':checked')) {
        // Filter is disabled - enable everything
        return [[], []];
    }

    /** @type {string[]} */
    const selectedVideoMimeTypes = Array.from($('input[name="vcodec"]:checked')).map((selectElement) => $(selectElement).val());

    /** @type {RTCRtpCodec[]} */
    const filteredVideoCodecs = selectedVideoMimeTypes.flatMap((mimeType) => {
        return allVCodecs.filter((codec) => codec.mimeType === mimeType);
    });

    /** @type {string[]} */
    const selectedAudioMimeTypes = Array.from($('input[name="acodec"]:checked')).map((selectElement) => $(selectElement).val());

    /** @type {RTCRtpCodec[]} */
    const filteredAudioCodecs = selectedAudioMimeTypes.flatMap((mimeType) => {
        return allACodecs.filter((codec) => codec.mimeType === mimeType);
    });

    console.log(
        `[${role}]`,
        `Filters: Video: ${selectedVideoMimeTypes.length ? selectedVideoMimeTypes : 'No filter'}, Audio: ${selectedAudioMimeTypes.length ? selectedAudioMimeTypes : 'No filter'}`,
    );

    console.debug(
        `[${role}]`,
        `All accepted codecs: Video:`,
        filteredVideoCodecs.length ? filteredVideoCodecs : 'ALL',
        'Audio:',
        filteredAudioCodecs.length ? filteredAudioCodecs : 'ALL',
    );

    return [filteredVideoCodecs, filteredAudioCodecs];
}

// The page is all setup. Hide the loading spinner and show the page content.
$('.loader').addClass('d-none');
$('#main').removeClass('d-none');
console.log('Page loaded');
