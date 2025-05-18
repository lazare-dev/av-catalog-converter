/**
 * Main Application JavaScript
 * Handles UI interactions and application logic
 */

// Application state
const AppState = {
    currentStep: 'upload', // 'upload', 'mapping', 'preview'
    jobId: null,
    file: null,
    company: '',
    mappings: {},
    sourceColumns: [],
    sampleData: {},
    previewData: [],
    requiredFields: ["SKU", "Short Description", "Manufacturer"]
};

// Standard fields in the correct order
const standardFields = [
    "SKU", "Short Description", "Long Description", "Model",
    "Category Group", "Category", "Manufacturer", "Manufacturer SKU",
    "Image URL", "Document Name", "Document URL", "Unit Of Measure",
    "Buy Cost", "Trade Price", "MSRP GBP", "MSRP USD", "MSRP EUR", "Discontinued"
];

// DOM Elements
const elements = {};

// Initialize the application
function initApp() {
    console.log('Initializing application...');

    // Initialize DOM elements
    initDOMElements();

    // Check API health
    API.checkHealth()
        .then(response => {
            console.log('API Health:', response);
        })
        .catch(error => {
            console.error('API Health Check Failed:', error);
            showError('API server is not available. Please check your connection.');
        });

    // Set up event listeners
    setupEventListeners();

    // Make sure the analyze button is disabled initially
    if (elements.analyzeButton) {
        elements.analyzeButton.disabled = true;
    }

    // Check for existing job in session storage
    const savedJobId = sessionStorage.getItem('jobId');
    if (savedJobId) {
        console.log('Found job ID in session storage:', savedJobId);
        AppState.jobId = savedJobId;

        // Try to verify the job ID by checking the mapping data endpoint
        fetch(`/api/get-mapping-data/${savedJobId}`)
            .then(response => {
                if (!response.ok) {
                    console.log('Job ID not valid for mapping data, trying analyze endpoint:', savedJobId);

                    // Try the analyze endpoint as a fallback
                    const formData = new FormData();
                    formData.append('job_id', savedJobId);

                    return fetch('/api/analyze', {
                        method: 'POST',
                        body: formData
                    }).then(analyzeResponse => {
                        if (!analyzeResponse.ok) {
                            console.error('Job ID invalid for both mapping and analyze:', savedJobId);
                            sessionStorage.removeItem('jobId');
                            AppState.jobId = null;
                            updateNavigationState();
                            return null;
                        }

                        console.log('Job ID valid for analyze endpoint:', savedJobId);
                        return analyzeResponse.json();
                    });
                }

                console.log('Job ID valid for mapping data:', savedJobId);
                return response.json();
            })
            .then(data => {
                if (data) {
                    console.log('Job data retrieved successfully');
                    updateNavigationState();
                }
            })
            .catch(error => {
                console.error('Error checking job validity:', error);
                // Keep the job ID in case the error is temporary
                updateNavigationState();
            });
    } else {
        // No job ID, disable mapping and preview steps
        updateNavigationState();
    }

    console.log('Application initialized');
}

// Update navigation state based on current app state
function updateNavigationState() {
    console.log('Updating navigation state...');
    console.log('Current state - Job ID:', AppState.jobId, 'Mappings:', Object.keys(AppState.mappings || {}).length);

    // Enable/disable navigation buttons based on state
    if (elements.navMapping) {
        elements.navMapping.disabled = !AppState.jobId;
        if (elements.navMapping.disabled) {
            elements.navMapping.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            elements.navMapping.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    if (elements.navPreview) {
        elements.navPreview.disabled = !AppState.jobId || !AppState.mappings || Object.keys(AppState.mappings).length === 0;
        if (elements.navPreview.disabled) {
            elements.navPreview.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            elements.navPreview.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }
}

// Initialize DOM elements
function initDOMElements() {
    console.log('Initializing DOM elements');

    // Navigation
    elements.navUpload = document.getElementById('nav-upload');
    elements.navMapping = document.getElementById('nav-mapping');
    elements.navPreview = document.getElementById('nav-preview');

    // Sections
    elements.sectionUpload = document.getElementById('section-upload');
    elements.sectionMapping = document.getElementById('section-mapping');
    elements.sectionPreview = document.getElementById('section-preview');

    // Upload section
    elements.companyName = document.getElementById('company-name');
    elements.fileUpload = document.getElementById('file-upload');
    elements.uploadContainer = document.getElementById('upload-container');
    elements.fileInfo = document.getElementById('file-info');
    elements.fileName = document.getElementById('file-name');
    elements.fileSize = document.getElementById('file-size');
    elements.uploadProgress = document.getElementById('upload-progress');
    elements.progressBar = document.getElementById('progress-bar');
    elements.progressText = document.getElementById('progress-text');
    elements.uploadError = document.getElementById('upload-error');
    elements.errorMessage = document.getElementById('error-message');
    elements.analyzeButton = document.getElementById('analyze-button');

    // Log file upload element for debugging
    console.log('File upload element:', elements.fileUpload);

    // Check if file upload element exists and has the correct attributes
    if (elements.fileUpload) {
        console.log('File upload element attributes:');
        console.log('- type:', elements.fileUpload.getAttribute('type'));
        console.log('- id:', elements.fileUpload.getAttribute('id'));
        console.log('- name:', elements.fileUpload.getAttribute('name'));
        console.log('- accept:', elements.fileUpload.getAttribute('accept'));
        console.log('- class:', elements.fileUpload.getAttribute('class'));
    } else {
        console.error('File upload element not found!');
        // Try to find it again with a different approach
        elements.fileUpload = document.querySelector('input[type="file"]');
        console.log('Trying alternative selector, result:', elements.fileUpload);
    }

    // Check if all required elements are found
    const requiredElements = [
        'fileUpload', 'uploadContainer', 'fileInfo', 'fileName',
        'fileSize', 'analyzeButton'
    ];

    const missingElements = requiredElements.filter(id => !elements[id]);

    if (missingElements.length > 0) {
        console.error('Missing required DOM elements:', missingElements);
        console.error('This may cause the application to malfunction.');

        // Try to recover from missing elements
        if (!elements.fileUpload) {
            console.log('Attempting to create file upload element');
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.id = 'file-upload';
            fileInput.name = 'file';
            fileInput.accept = '.csv,.xlsx,.xls,.pdf,.json,.xml';
            fileInput.style.display = 'block';
            fileInput.style.margin = '20px auto';

            if (elements.uploadContainer) {
                elements.uploadContainer.appendChild(fileInput);
                elements.fileUpload = fileInput;
                console.log('Created new file upload element:', fileInput);
            }
        }
    } else {
        console.log('All required DOM elements initialized successfully');
    }
}

// Set up event listeners
function setupEventListeners() {
    console.log('Setting up event listeners');

    try {
        // Company name input
        if (elements.companyName) {
            console.log('Adding input event listener to company name field');
            elements.companyName.addEventListener('input', function() {
                // Enable/disable analyze button based on company name and file selection
                if (elements.analyzeButton) {
                    elements.analyzeButton.disabled = !this.value.trim() || !AppState.file;
                }
            });
        }

        // File upload via input
        if (elements.fileUpload) {
            console.log('Adding change event listener to file upload input');

            // Remove any existing event listeners first
            const newFileUpload = elements.fileUpload.cloneNode(true);
            elements.fileUpload.parentNode.replaceChild(newFileUpload, elements.fileUpload);
            elements.fileUpload = newFileUpload;

            // Add the event listener
            elements.fileUpload.addEventListener('change', function(event) {
                console.log('File input change event triggered:', event);
                handleFileSelect(event);
            });

            // Add a direct click handler to the browse files link
            const browseLink = document.querySelector('label[for="file-upload"]');
            if (browseLink) {
                console.log('Adding click event listener to browse files link');
                browseLink.addEventListener('click', function(e) {
                    console.log('Browse files link clicked');
                    // Explicitly trigger click on file input for better cross-browser compatibility
                    setTimeout(() => {
                        elements.fileUpload.click();
                    }, 100);
                });
            } else {
                console.error('Browse files link not found');
                // Create a browse button if the link is missing
                if (elements.uploadContainer) {
                    const browseButton = document.createElement('button');
                    browseButton.textContent = 'Browse Files';
                    browseButton.className = 'px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none';
                    browseButton.addEventListener('click', function() {
                        elements.fileUpload.click();
                    });
                    elements.uploadContainer.appendChild(browseButton);
                    console.log('Created browse button as fallback');
                }
            }
        } else {
            console.error('File upload element not found');
        }

        // Drag and drop
        if (elements.uploadContainer) {
            console.log('Adding drag and drop event listeners to upload container');

            // Remove any existing event listeners first
            const newUploadContainer = elements.uploadContainer.cloneNode(true);
            elements.uploadContainer.parentNode.replaceChild(newUploadContainer, elements.uploadContainer);
            elements.uploadContainer = newUploadContainer;

            // Re-find the file upload element if it was inside the container
            if (!elements.fileUpload) {
                elements.fileUpload = elements.uploadContainer.querySelector('input[type="file"]');
                if (elements.fileUpload) {
                    elements.fileUpload.addEventListener('change', handleFileSelect);
                }
            }

            // Add the event listeners
            elements.uploadContainer.addEventListener('dragover', handleDragOver);
            elements.uploadContainer.addEventListener('dragleave', handleDragLeave);
            elements.uploadContainer.addEventListener('drop', handleFileDrop);
            elements.uploadContainer.addEventListener('click', function(e) {
                console.log('Upload container clicked');
                // Only trigger file input click if the click wasn't on a child element
                if (e.target === elements.uploadContainer && elements.fileUpload) {
                    elements.fileUpload.click();
                }
            });
        } else {
            console.error('Upload container element not found');
        }

        // Analyze button
        if (elements.analyzeButton) {
            console.log('Adding click event listener to analyze button');

            // Remove any existing event listeners first
            const newAnalyzeButton = elements.analyzeButton.cloneNode(true);
            elements.analyzeButton.parentNode.replaceChild(newAnalyzeButton, elements.analyzeButton);
            elements.analyzeButton = newAnalyzeButton;

            // Add the event listener
            elements.analyzeButton.addEventListener('click', handleAnalyze);

            // Enable the button if we have a file selected
            if (AppState.file) {
                elements.analyzeButton.disabled = false;
            }
        } else {
            console.error('Analyze button element not found');
        }

        // Navigation
        if (elements.navUpload && elements.navMapping && elements.navPreview) {
            console.log('Adding click event listeners to navigation buttons');
            elements.navUpload.addEventListener('click', () => navigateTo('upload'));
            elements.navMapping.addEventListener('click', () => navigateTo('mapping'));
            elements.navPreview.addEventListener('click', () => navigateTo('preview'));
        } else {
            console.error('Navigation elements not found');
        }

        console.log('Event listeners set up successfully');
    } catch (error) {
        console.error('Error setting up event listeners:', error);
    }
}

// Handle file selection via input
function handleFileSelect(event) {
    console.log('File selected via input:', event);

    // Make sure we have the event and it has files
    if (!event || !event.target || !event.target.files || event.target.files.length === 0) {
        console.error('Invalid file selection event or no files selected');

        // Try to get files from the file input element directly
        if (elements.fileUpload && elements.fileUpload.files && elements.fileUpload.files.length > 0) {
            console.log('Found files in the file input element:', elements.fileUpload.files);
            const file = elements.fileUpload.files[0];
            processSelectedFile(file);
            return;
        }

        return;
    }

    const file = event.target.files[0];
    console.log('Selected file:', file);
    processSelectedFile(file);
}

// Process the selected file
function processSelectedFile(file) {
    if (file) {
        console.log('Processing selected file:', file.name, 'Size:', file.size);

        // Clear any existing job ID when a new file is selected
        clearJobId();
        setSelectedFile(file);

        // Make sure the analyze button is enabled only if company name is provided
        if (elements.analyzeButton && elements.companyName) {
            console.log('Checking company name for button enablement');
            const companyName = elements.companyName.value.trim();
            elements.analyzeButton.disabled = !companyName;
        }
    } else {
        console.error('No file to process');
    }
}

// Clear job ID from state and session storage
function clearJobId() {
    // If we have an existing job ID, clear it
    if (AppState.jobId) {
        console.log(`Clearing existing job ID: ${AppState.jobId}`);
        AppState.jobId = null;
        sessionStorage.removeItem('jobId');

        // Also clear any existing mappings
        AppState.mappings = {};
        AppState.sourceColumns = [];
        AppState.sampleData = {};
    }
}

// Handle drag over event
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadContainer.classList.add('border-blue-500');
}

// Handle drag leave event
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadContainer.classList.remove('border-blue-500');
}

// Handle file drop event
function handleFileDrop(event) {
    console.log('File dropped:', event);

    event.preventDefault();
    event.stopPropagation();

    // Remove the highlight from the drop zone
    if (elements.uploadContainer) {
        elements.uploadContainer.classList.remove('border-blue-500');
    }

    // Make sure we have files in the drop event
    if (!event.dataTransfer || !event.dataTransfer.files || event.dataTransfer.files.length === 0) {
        console.error('Invalid drop event or no files dropped');
        return;
    }

    const file = event.dataTransfer.files[0];
    console.log('Dropped file:', file);

    if (file) {
        // If we have a file input element, set its files property
        if (elements.fileUpload) {
            try {
                // Create a DataTransfer object to set the files property
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                elements.fileUpload.files = dataTransfer.files;
                console.log('Set file input files property:', elements.fileUpload.files);
            } catch (error) {
                console.error('Error setting file input files property:', error);
            }
        }

        // Process the file
        processSelectedFile(file);
    } else {
        console.error('No file dropped or drop event did not contain files');
    }
}

// Set the selected file and update UI
function setSelectedFile(file) {
    console.log('Setting selected file:', file);

    // Store the file in the application state
    AppState.file = file;

    try {
        // Log elements for debugging
        console.log('File name element:', elements.fileName);
        console.log('File size element:', elements.fileSize);
        console.log('File info element:', elements.fileInfo);
        console.log('Upload container element:', elements.uploadContainer);
        console.log('Analyze button element:', elements.analyzeButton);

        // Update file info display
        if (elements.fileName && elements.fileSize) {
            elements.fileName.textContent = file.name;
            elements.fileSize.textContent = formatFileSize(file.size);

            // Show file info
            if (elements.fileInfo) {
                elements.fileInfo.classList.remove('hidden');
            }

            // Enable analyze button
            if (elements.analyzeButton) {
                elements.analyzeButton.disabled = false;
            }

            // Add a visual indicator to the upload container
            if (elements.uploadContainer) {
                elements.uploadContainer.classList.add('border-green-500');
                elements.uploadContainer.classList.add('bg-green-50');

                // Update the upload container text to show the selected file
                const uploadText = elements.uploadContainer.querySelector('p');
                if (uploadText) {
                    uploadText.innerHTML = `Selected file: <strong>${file.name}</strong>`;
                }
            }

            console.log('File info updated successfully');
        } else {
            console.error('File name or file size elements not found');
        }

        // Hide any previous errors
        if (elements.uploadError) {
            elements.uploadError.classList.add('hidden');
        }
    } catch (error) {
        console.error('Error updating file info:', error);
        alert('Error updating file information. Please try again or refresh the page.');
    }
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle analyze button click
async function handleAnalyze() {
    if (!AppState.file) {
        showError('Please select a file first.');
        return;
    }

    // Get company name
    AppState.company = elements.companyName.value.trim() || 'Unknown';

    console.log('Starting file upload and analysis process');
    console.log('File:', AppState.file.name, 'Size:', AppState.file.size, 'Company:', AppState.company);

    try {
        // Show progress
        elements.uploadProgress.classList.remove('hidden');
        elements.analyzeButton.disabled = true;

        // Clear any existing job ID
        AppState.jobId = null;
        sessionStorage.removeItem('jobId');

        // Create FormData object directly
        const formData = new FormData();
        formData.append('file', AppState.file);
        formData.append('company', AppState.company);

        console.log('Uploading file directly via fetch API');

        // Upload the file directly using fetch
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            throw new Error(errorData.error || 'Upload failed');
        }

        const uploadResult = await uploadResponse.json();
        console.log('Upload Result:', uploadResult);

        if (!uploadResult.success || !uploadResult.job_id) {
            throw new Error('Upload response did not contain a valid job ID');
        }

        // Store job ID
        AppState.jobId = uploadResult.job_id;
        sessionStorage.setItem('jobId', AppState.jobId);

        // Update progress
        updateProgressBar(100);
        elements.progressText.textContent = 'Analyzing file...';

        console.log('File uploaded successfully, job ID:', AppState.jobId);
        console.log('Now analyzing file...');

        // Create FormData for analysis
        const analysisFormData = new FormData();
        analysisFormData.append('job_id', AppState.jobId);

        // Also include the file in case the job ID is not found
        analysisFormData.append('file', AppState.file);
        analysisFormData.append('company', AppState.company);

        console.log('Sending analysis request with job ID:', AppState.jobId);
        console.log('Also including file as fallback:', AppState.file.name);

        // Analyze the file
        const analysisResponse = await fetch('/api/analyze', {
            method: 'POST',
            body: analysisFormData
        });

        if (!analysisResponse.ok) {
            const errorData = await analysisResponse.json();
            console.error('Analysis failed:', errorData);

            // If the job ID is invalid, try again with a direct file upload
            if (errorData.error === 'Invalid job ID' || errorData.error === 'No file or valid job ID provided') {
                console.log('Job ID invalid, trying direct file upload for analysis');

                // Create a new FormData with the file
                const directFormData = new FormData();
                directFormData.append('file', AppState.file);

                // Try direct file upload for analysis
                const directResponse = await fetch('/api/analyze', {
                    method: 'POST',
                    body: directFormData
                });

                if (!directResponse.ok) {
                    const directErrorData = await directResponse.json();
                    console.error('Direct analysis failed:', directErrorData);
                    throw new Error(directErrorData.error || 'Analysis failed');
                }

                const directResult = await directResponse.json();
                console.log('Direct Analysis Result:', directResult);

                if (!directResult.success) {
                    console.error('Direct analysis not successful:', directResult);
                    throw new Error(directResult.error || 'Analysis failed');
                }

                // Update job ID from direct analysis
                if (directResult.job_id) {
                    console.log(`Updating job ID to ${directResult.job_id} from direct analysis`);
                    AppState.jobId = directResult.job_id;
                    sessionStorage.setItem('jobId', AppState.jobId);
                    return directResult;
                }
            } else {
                throw new Error(errorData.error || 'Analysis failed');
            }
        }

        const analysisResult = await analysisResponse.json();
        console.log('Analysis Result:', analysisResult);

        if (!analysisResult.success) {
            console.error('Analysis not successful:', analysisResult);
            throw new Error(analysisResult.error || 'Analysis failed');
        }

        // Make sure we have the correct job ID from the response
        if (analysisResult.job_id && analysisResult.job_id !== AppState.jobId) {
            console.log(`Updating job ID from ${AppState.jobId} to ${analysisResult.job_id}`);
            AppState.jobId = analysisResult.job_id;
            sessionStorage.setItem('jobId', AppState.jobId);
        }

        // Wait a moment for the backend to process the file
        console.log('Waiting for backend processing to complete...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Hide progress
        elements.uploadProgress.classList.add('hidden');

        // Update navigation state
        updateNavigationState();

        // Navigate to mapping step
        navigateTo('mapping');
    } catch (error) {
        console.error('Analysis failed:', error);
        showError(error.message || 'Failed to analyze the file. Please try again.');
        elements.uploadProgress.classList.add('hidden');
        elements.analyzeButton.disabled = false;
    }
}

// Update progress bar
function updateProgressBar(percent) {
    elements.progressBar.style.width = `${percent}%`;
    elements.progressText.textContent = `Uploading... ${percent}%`;
}

// Show error message
function showError(message) {
    if (elements.errorMessage && elements.uploadError) {
        elements.errorMessage.textContent = message;
        elements.uploadError.classList.remove('hidden');
    } else {
        console.error('Error:', message);
        // If elements are not initialized yet, we'll just log the error
    }
}

// Navigate to a step
function navigateTo(step) {
    console.log('Navigating to step:', step);

    // Validate step
    if (!['upload', 'mapping', 'preview'].includes(step)) {
        console.error('Invalid step:', step);
        return;
    }

    // Check if we can navigate to the requested step
    if (step === 'mapping' && !AppState.jobId) {
        console.error('Cannot navigate to mapping step: No job ID');
        showError('Please upload and analyze a file first before proceeding to mapping.');
        return;
    }

    if (step === 'preview' && (!AppState.jobId || !AppState.mappings || Object.keys(AppState.mappings).length === 0)) {
        console.error('Cannot navigate to preview step: No job ID or mappings');
        showError('Please complete the mapping step before proceeding to preview.');
        return;
    }

    // Update current step
    AppState.currentStep = step;

    // Update navigation
    elements.navUpload.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
    elements.navUpload.classList.add('text-gray-500');
    elements.navMapping.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
    elements.navMapping.classList.add('text-gray-500');
    elements.navPreview.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
    elements.navPreview.classList.add('text-gray-500');

    // Hide all sections
    elements.sectionUpload.classList.add('hidden');
    elements.sectionMapping.classList.add('hidden');
    elements.sectionPreview.classList.add('hidden');

    // Show active section and update navigation
    if (step === 'upload') {
        elements.sectionUpload.classList.remove('hidden');
        elements.navUpload.classList.remove('text-gray-500');
        elements.navUpload.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');

        // Enable/disable navigation buttons based on state
        elements.navUpload.disabled = false;
        elements.navMapping.disabled = !AppState.jobId;
        elements.navPreview.disabled = !AppState.jobId || !AppState.mappings || Object.keys(AppState.mappings).length === 0;
    } else if (step === 'mapping') {
        elements.sectionMapping.classList.remove('hidden');
        elements.navMapping.classList.remove('text-gray-500');
        elements.navMapping.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');

        // Enable/disable navigation buttons
        elements.navUpload.disabled = false;
        elements.navMapping.disabled = false;
        elements.navPreview.disabled = !AppState.mappings || Object.keys(AppState.mappings).length === 0;

        // Load mapping data
        loadMappingStep();
    } else if (step === 'preview') {
        elements.sectionPreview.classList.remove('hidden');
        elements.navPreview.classList.remove('text-gray-500');
        elements.navPreview.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');

        // Enable all navigation buttons
        elements.navUpload.disabled = false;
        elements.navMapping.disabled = false;
        elements.navPreview.disabled = false;

        // Load preview data
        loadPreviewStep();
    }

    console.log('Navigation complete');
}

// Load mapping step
async function loadMappingStep() {
    console.log('Loading mapping step...');

    // Check if we have a job ID
    if (!AppState.jobId) {
        // Check if there's a job ID in session storage
        const savedJobId = sessionStorage.getItem('jobId');
        if (savedJobId) {
            console.log('Found job ID in session storage:', savedJobId);
            AppState.jobId = savedJobId;
        } else {
            console.error('No job ID available');
            showMappingError('No Job ID Available', 'Please upload a file first to generate a job ID.');
            return;
        }
    }

    console.log('Using job ID:', AppState.jobId);

    try {
        // Show loading state
        elements.sectionMapping.innerHTML = `
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4">Field Mapping</h2>
                <div class="flex items-center justify-center py-8">
                    <svg class="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span class="ml-3 text-gray-600">Loading mapping data...</span>
                </div>
            </div>
        `;

        // Maximum number of retries
        const maxRetries = 3;
        let retryCount = 0;
        let mappingData = null;
        let lastError = null;

        // Try to get mapping data with retries
        while (retryCount < maxRetries) {
            try {
                console.log(`Getting mapping data (attempt ${retryCount + 1} of ${maxRetries})...`);

                // Get mapping data
                mappingData = await API.getMappingData(AppState.jobId);
                console.log('Mapping Data:', mappingData);

                // Check for both source_columns and columns fields
                if (!mappingData.source_columns && mappingData.columns) {
                    console.log('Using columns field as source_columns');
                    mappingData.source_columns = mappingData.columns;
                }

                if (mappingData &&
                    ((mappingData.source_columns && mappingData.source_columns.length > 0) ||
                     (mappingData.columns && mappingData.columns.length > 0))) {
                    // Success! Break out of the retry loop
                    break;
                } else {
                    console.warn('Received empty or invalid mapping data, retrying...');
                    lastError = new Error('Received invalid mapping data from the server');
                }
            } catch (error) {
                console.error(`Attempt ${retryCount + 1} failed:`, error);
                lastError = error;

                // If it's a 404 or "Invalid job ID" error, try to reanalyze the file
                if (error.message && (
                    error.message.includes('Invalid job ID') ||
                    error.message.includes('Job not found') ||
                    error.message.includes('404')
                )) {
                    console.log('Job ID invalid, trying to reanalyze');

                    try {
                        // Try to use the analyze endpoint with the job ID
                        const formData = new FormData();
                        formData.append('job_id', AppState.jobId);

                        // Also include the file if we have it
                        if (AppState.file) {
                            console.log('Including file in reanalysis request:', AppState.file.name);
                            formData.append('file', AppState.file);

                            if (AppState.company) {
                                formData.append('company', AppState.company);
                            }
                        }

                        console.log('Sending reanalysis request with job ID:', AppState.jobId);
                        const analyzeResponse = await fetch('/api/analyze', {
                            method: 'POST',
                            body: formData
                        });

                        if (analyzeResponse.ok) {
                            console.log('Successfully reanalyzed with job ID:', AppState.jobId);
                            const analyzeResult = await analyzeResponse.json();

                            // If we got a new job ID, update it
                            if (analyzeResult.job_id && analyzeResult.job_id !== AppState.jobId) {
                                console.log(`Updating job ID from ${AppState.jobId} to ${analyzeResult.job_id}`);
                                AppState.jobId = analyzeResult.job_id;
                                sessionStorage.setItem('jobId', AppState.jobId);

                                // Try to get mapping data again with the new job ID
                                continue;
                            }
                        } else {
                            // If reanalysis failed, don't retry
                            break;
                        }
                    } catch (analyzeError) {
                        console.error('Error reanalyzing:', analyzeError);
                        // If reanalysis failed with an error, don't retry
                        break;
                    }
                }
            }

            // Wait before retrying
            console.log(`Waiting before retry ${retryCount + 1}...`);
            await new Promise(resolve => setTimeout(resolve, 2000));
            retryCount++;
        }

        // If we didn't get valid mapping data after all retries, throw the last error
        if (!mappingData ||
            ((!mappingData.source_columns || mappingData.source_columns.length === 0) &&
             (!mappingData.columns || mappingData.columns.length === 0))) {
            throw lastError || new Error('Failed to get mapping data after multiple attempts');
        }

        // If source_columns doesn't exist but columns does, use columns as source_columns
        if (!mappingData.source_columns && mappingData.columns) {
            console.log('Using columns field as source_columns');
            mappingData.source_columns = mappingData.columns;
        }

        // Store mapping data in app state
        console.log('Detailed mapping data inspection:');
        console.log('mappingData object keys:', Object.keys(mappingData));
        console.log('mappingData.mappings:', mappingData.mappings);
        console.log('mappingData.field_mappings:', mappingData.field_mappings);
        console.log('mappingData.analyze_response:', mappingData.analyze_response);

        // Check if field_mappings exists but mappings doesn't
        if (!mappingData.mappings && mappingData.field_mappings) {
            console.log('Using field_mappings instead of mappings');
            mappingData.mappings = mappingData.field_mappings;
        }

        // Check if analyze_response.field_mappings exists but mappings doesn't
        if (!mappingData.mappings && mappingData.analyze_response && mappingData.analyze_response.field_mappings) {
            console.log('Using analyze_response.field_mappings');

            // Convert from the analyze response format to the format expected by the frontend
            const analyzeMappings = mappingData.analyze_response.field_mappings;
            mappingData.mappings = {};

            for (const [targetField, mappingInfo] of Object.entries(analyzeMappings)) {
                if (typeof mappingInfo === 'object' && mappingInfo.column) {
                    mappingData.mappings[targetField] = mappingInfo.column;
                } else if (typeof mappingInfo === 'string') {
                    mappingData.mappings[targetField] = mappingInfo;
                }
            }

            console.log('Converted mappings:', mappingData.mappings);
        }

        // Initialize mappings if it doesn't exist
        if (!mappingData.mappings) {
            mappingData.mappings = {};
        }

        // FORCE the company name into the Manufacturer field
        if (AppState.company) {
            console.log('FORCING company name into Manufacturer field:', AppState.company);
            mappingData.mappings['Manufacturer'] = `__DIRECT_VALUE__:${AppState.company}`;
        }

        console.log('Final mappings after forcing company name:', mappingData.mappings);

        AppState.mappings = mappingData.mappings || {};
        AppState.sourceColumns = mappingData.source_columns || [];
        AppState.sampleData = mappingData.sample_data || {};
        if (mappingData.required_fields) {
            AppState.requiredFields = mappingData.required_fields;
        }

        console.log('Mapping data loaded successfully');
        console.log('Source columns:', AppState.sourceColumns);
        console.log('Mappings:', AppState.mappings);

        // Render mapping interface
        renderMappingInterface();
    } catch (error) {
        console.error('Failed to load mapping data:', error);

        // Check if it's an invalid job ID error
        if (error.message && (
            error.message.includes('Invalid job ID') ||
            error.message.includes('Job not found') ||
            error.message.includes('404')
        )) {
            // Clear the invalid job ID from session storage
            console.log('Clearing invalid job ID from session storage');
            sessionStorage.removeItem('jobId');
            AppState.jobId = null;

            showMappingError('Invalid Job ID',
                'The job ID is invalid or has expired. Please upload your file again.',
                ['The server may have been restarted since your last upload',
                 'Job data may have been deleted or expired',
                 'There might have been an error during file analysis']);
        } else if (error.message && error.message.includes('not ready')) {
            // Job exists but is not ready
            showMappingError('Job Not Ready',
                'The file is still being processed. Please wait a moment and try again.',
                ['The analysis process may take some time for large files',
                 'You can try refreshing the page or clicking "Retry" in a few seconds']);
        } else {
            // Other error
            showMappingError('Error Loading Mapping Data',
                error.message || 'Failed to load mapping data. Please try again.',
                ['The server might be busy processing your file',
                 'Large files may take longer to process',
                 'You can try refreshing the page or clicking "Retry" in a few seconds']);
        }
    }
}

// Show mapping error with custom message
function showMappingError(title, message, explanations = []) {
    let explanationsHtml = '';
    if (explanations && explanations.length > 0) {
        explanationsHtml = '<ul class="mt-2 text-sm list-disc list-inside">';
        explanations.forEach(explanation => {
            explanationsHtml += `<li>${explanation}</li>`;
        });
        explanationsHtml += '</ul>';
    }

    elements.sectionMapping.innerHTML = `
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4">Field Mapping</h2>
            <div class="bg-red-50 p-4 rounded-lg text-red-700 mb-4">
                <div class="flex">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                        <p class="font-medium">${title}</p>
                        <p class="text-sm">${message}</p>
                        ${explanationsHtml}
                    </div>
                </div>
            </div>
            <div class="flex justify-between">
                <button onclick="navigateTo('upload')" class="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50">
                    Back to Upload
                </button>
                ${title !== 'No Job ID Available' ?
                `<button onclick="loadMappingStep()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
                    Retry
                </button>` : ''}
            </div>
        </div>
    `;
}

// Render mapping interface
function renderMappingInterface() {
    // Create mapping interface HTML
    let mappingHtml = `
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4">Field Mapping</h2>
            <p class="mb-4 text-gray-600">Review and adjust the field mappings for your catalog data.</p>

            <div class="overflow-x-auto mb-6">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Standard Field</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source Column</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sample Data</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Required</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200" id="mapping-tbody">
    `;

    // Add rows for each standard field
    standardFields.forEach(field => {
        const isRequired = AppState.requiredFields.includes(field);
        const currentMapping = AppState.mappings[field] || '';

        mappingHtml += `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm font-medium text-gray-900">
                        ${field} ${isRequired ? '<span class="text-red-500">*</span>' : ''}
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    ${currentMapping && currentMapping.startsWith('__DIRECT_VALUE__:') ?
                        `<input type="text" id="mapping-${field}" class="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                value="${currentMapping.split(':')[1]}" data-is-direct="true">` :
                        `<select id="mapping-${field}" class="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm" onchange="updateSampleData('${field}', this.value)">
                            <option value="">-- Select Source Column --</option>
                            ${AppState.sourceColumns.map(column => `
                                <option value="${column}" ${currentMapping === column ? 'selected' : ''}>${column}</option>
                            `).join('')}
                        </select>`
                    }
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div id="sample-${field}" class="text-sm text-gray-500 max-w-xs truncate">
                        ${currentMapping ?
                            (currentMapping.startsWith('__DIRECT_VALUE__:') ?
                                currentMapping.split(':')[1] :
                                getSampleDataDisplay(currentMapping)) :
                            '[No data]'}
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-500">
                        ${isRequired ? 'Yes' : 'No'}
                    </div>
                </td>
            </tr>
        `;
    });

    // Close table and add buttons
    mappingHtml += `
                    </tbody>
                </table>
            </div>

            <div id="validation-results" class="hidden mb-6 p-4 rounded-lg"></div>

            <div class="flex justify-between">
                <button onclick="resetMappings()" class="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50">
                    Reset to Suggested Mappings
                </button>
                <button onclick="validateMappings()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
                    Validate and Continue
                </button>
            </div>
        </div>
    `;

    // Update the mapping section
    elements.sectionMapping.innerHTML = mappingHtml;
}

// Get sample data display
function getSampleDataDisplay(column) {
    if (!AppState.sampleData[column]) {
        return '[No sample data]';
    }

    const data = AppState.sampleData[column];

    if (Array.isArray(data)) {
        const validData = data.filter(item => item !== null && item !== undefined && item !== '');
        if (validData.length === 0) return '[No valid sample data]';

        return validData.slice(0, 3).join(', ');
    }

    return String(data);
}

// Update sample data display
function updateSampleData(field, column) {
    const sampleCell = document.getElementById(`sample-${field}`);
    if (!sampleCell) return;

    if (!column) {
        sampleCell.textContent = '[No data]';
        return;
    }

    sampleCell.textContent = getSampleDataDisplay(column);
}

// Reset mappings to original suggestions
function resetMappings() {
    standardFields.forEach(field => {
        const select = document.getElementById(`mapping-${field}`);
        if (select && AppState.mappings[field]) {
            select.value = AppState.mappings[field];
            updateSampleData(field, AppState.mappings[field]);
        }
    });

    // Hide validation results
    const validationResults = document.getElementById('validation-results');
    if (validationResults) {
        validationResults.classList.add('hidden');
    }
}

// Validate mappings
async function validateMappings() {
    try {
        // Collect current mappings
        const currentMappings = {};
        standardFields.forEach(field => {
            const element = document.getElementById(`mapping-${field}`);
            if (!element) return;

            // Check if it's a direct value input or a select
            const isDirectValue = element.tagName.toLowerCase() === 'input' && element.getAttribute('data-is-direct') === 'true';

            if (isDirectValue) {
                // For direct values, use the special prefix
                if (element.value.trim()) {
                    currentMappings[field] = `__DIRECT_VALUE__:${element.value.trim()}`;
                }
            } else if (element.tagName.toLowerCase() === 'select' && element.value) {
                // For select elements, use the selected value
                currentMappings[field] = element.value;
            }
        });

        // Show loading state
        const validationResults = document.getElementById('validation-results');
        validationResults.classList.remove('hidden');
        validationResults.classList.remove('bg-red-50', 'text-red-700', 'bg-green-50', 'text-green-700');
        validationResults.classList.add('bg-blue-50', 'text-blue-700');
        validationResults.innerHTML = `
            <div class="flex items-center">
                <svg class="animate-spin h-5 w-5 mr-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Validating mappings...</span>
            </div>
        `;

        // Validate mappings
        const validationResult = await API.validateMapping(AppState.jobId, currentMappings);
        console.log('Validation Result:', validationResult);

        // Update validation results display
        if (validationResult.success) {
            // Store validated mappings
            AppState.mappings = currentMappings;

            // Show success message
            validationResults.classList.remove('bg-blue-50', 'text-blue-700');
            validationResults.classList.add('bg-green-50', 'text-green-700');
            validationResults.innerHTML = `
                <div class="flex">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                    </svg>
                    <div>
                        <p class="font-medium">Validation Successful</p>
                        <p class="text-sm">Your mappings are valid. Click Continue to preview the data.</p>
                    </div>
                </div>
                <div class="mt-4 text-right">
                    <button onclick="navigateTo('preview')" class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50">
                        Continue to Preview
                    </button>
                </div>
            `;
        } else {
            // Show error message
            validationResults.classList.remove('bg-blue-50', 'text-blue-700');
            validationResults.classList.add('bg-red-50', 'text-red-700');

            let errorHtml = `
                <div class="flex">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                        <p class="font-medium">Validation Failed</p>
                        <p class="text-sm">Please fix the following issues:</p>
                        <ul class="mt-2 text-sm list-disc list-inside">
            `;

            if (validationResult.issues.missing_required && validationResult.issues.missing_required.length > 0) {
                errorHtml += `<li>Missing required fields: ${validationResult.issues.missing_required.join(', ')}</li>`;
            }

            if (validationResult.issues.unknown_fields && validationResult.issues.unknown_fields.length > 0) {
                errorHtml += `<li>Unknown target fields: ${validationResult.issues.unknown_fields.join(', ')}</li>`;
            }

            if (validationResult.issues.duplicate_mappings && validationResult.issues.duplicate_mappings.length > 0) {
                errorHtml += `<li>Duplicate mappings: ${validationResult.issues.duplicate_mappings.join(', ')}</li>`;
            }

            errorHtml += `
                        </ul>
                    </div>
                </div>
            `;

            validationResults.innerHTML = errorHtml;
        }
    } catch (error) {
        console.error('Validation failed:', error);

        // Show error message
        const validationResults = document.getElementById('validation-results');
        validationResults.classList.remove('hidden', 'bg-blue-50', 'text-blue-700');
        validationResults.classList.add('bg-red-50', 'text-red-700');
        validationResults.innerHTML = `
            <div class="flex">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                    <p class="font-medium">Validation Error</p>
                    <p class="text-sm">${error.message || 'An error occurred during validation. Please try again.'}</p>
                </div>
            </div>
        `;
    }
}

// Load preview step
async function loadPreviewStep() {
    // Check if we have a job ID and mappings
    if (!AppState.jobId) {
        // Check if there's a job ID in session storage
        const savedJobId = sessionStorage.getItem('jobId');
        if (savedJobId) {
            AppState.jobId = savedJobId;
        } else {
            showPreviewError('No Job ID Available', 'Please upload a file first to generate a job ID.');
            return;
        }
    }

    // Check if we have mappings
    if (Object.keys(AppState.mappings).length === 0) {
        showPreviewError('No Mappings Available', 'Please complete the field mapping step first.');
        return;
    }

    try {
        // Show loading state
        elements.sectionPreview.innerHTML = `
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4">Preview & Export</h2>
                <div class="flex items-center justify-center py-8">
                    <svg class="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span class="ml-3 text-gray-600">Processing data...</span>
                </div>
            </div>
        `;

        // Process the file with mappings
        const processResult = await API.processFile(AppState.jobId, AppState.mappings, 'csv');
        console.log('Process Result:', processResult);

        // Render preview interface
        renderPreviewInterface(processResult);
    } catch (error) {
        console.error('Failed to process data:', error);

        // Check if it's an invalid job ID error
        if (error.message && error.message.includes('Invalid job ID')) {
            // Clear the invalid job ID from session storage
            sessionStorage.removeItem('jobId');
            AppState.jobId = null;

            showPreviewError('Invalid Job ID',
                'The job ID is invalid or has expired. Please upload your file again.',
                ['The server may have been restarted since your last upload',
                 'Job data may have been deleted or expired',
                 'There might have been an error during file analysis']);
        } else {
            showPreviewError('Error Processing Data', error.message || 'Failed to process data. Please try again.');
        }
    }
}

// Show preview error with custom message
function showPreviewError(title, message, explanations = []) {
    let explanationsHtml = '';
    if (explanations && explanations.length > 0) {
        explanationsHtml = '<ul class="mt-2 text-sm list-disc list-inside">';
        explanations.forEach(explanation => {
            explanationsHtml += `<li>${explanation}</li>`;
        });
        explanationsHtml += '</ul>';
    }

    elements.sectionPreview.innerHTML = `
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4">Preview & Export</h2>
            <div class="bg-red-50 p-4 rounded-lg text-red-700 mb-4">
                <div class="flex">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                        <p class="font-medium">${title}</p>
                        <p class="text-sm">${message}</p>
                        ${explanationsHtml}
                    </div>
                </div>
            </div>
            <div class="flex justify-between">
                <button onclick="navigateTo('mapping')" class="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50">
                    Back to Mapping
                </button>
                ${title !== 'No Job ID Available' && title !== 'No Mappings Available' ?
                `<button onclick="loadPreviewStep()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
                    Retry
                </button>` : ''}
            </div>
        </div>
    `;
}

// Render preview interface
function renderPreviewInterface(processResult) {
    // Store preview data
    AppState.previewData = processResult.preview_data || [];

    // Get column headers from the first row
    const headers = AppState.previewData.length > 0 ? Object.keys(AppState.previewData[0]) : [];

    // Create preview interface HTML
    let previewHtml = `
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl font-semibold mb-4">Preview & Export</h2>
            <p class="mb-4 text-gray-600">Review your standardized catalog data and export when ready.</p>

            <div class="overflow-x-auto mb-6">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            ${headers.map(header => `
                                <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">${header}</th>
                            `).join('')}
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
    `;

    // Add rows for preview data (limit to 10 rows)
    const previewRows = AppState.previewData.slice(0, 10);
    previewRows.forEach(row => {
        previewHtml += `
            <tr>
                ${headers.map(header => `
                    <td class="px-4 py-3 whitespace-nowrap">
                        <div class="text-sm text-gray-900 truncate max-w-xs">${row[header] !== null && row[header] !== undefined ? row[header] : ''}</div>
                    </td>
                `).join('')}
            </tr>
        `;
    });

    // Close table and add export options
    previewHtml += `
                    </tbody>
                </table>
            </div>

            <div class="bg-gray-50 p-4 rounded-lg mb-6">
                <h3 class="text-lg font-medium mb-2">Export Options</h3>
                <div class="flex items-center space-x-4">
                    <div class="flex items-center">
                        <input type="radio" id="format-csv" name="format" value="csv" checked class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                        <label for="format-csv" class="ml-2 block text-sm text-gray-700">CSV</label>
                    </div>
                    <div class="flex items-center">
                        <input type="radio" id="format-excel" name="format" value="excel" class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                        <label for="format-excel" class="ml-2 block text-sm text-gray-700">Excel</label>
                    </div>
                    <div class="flex items-center">
                        <input type="radio" id="format-json" name="format" value="json" class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded">
                        <label for="format-json" class="ml-2 block text-sm text-gray-700">JSON</label>
                    </div>
                </div>
            </div>

            <div class="flex justify-between">
                <button onclick="navigateTo('mapping')" class="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50">
                    Back to Mapping
                </button>
                <button onclick="exportData()" class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50">
                    Export Data
                </button>
            </div>
        </div>
    `;

    // Update the preview section
    elements.sectionPreview.innerHTML = previewHtml;
}

// Export data
function exportData() {
    // Get selected format
    const formatRadios = document.getElementsByName('format');
    let selectedFormat = 'csv';

    for (const radio of formatRadios) {
        if (radio.checked) {
            selectedFormat = radio.value;
            break;
        }
    }

    // Check if we have a valid job ID
    if (!AppState.jobId) {
        console.error('No job ID available for export');
        alert('Error: No job ID available. Please upload and process a file first.');
        return;
    }

    // Get download URL with format
    const downloadUrl = API.getDownloadUrl(AppState.jobId, selectedFormat);
    console.log(`Exporting data with format: ${selectedFormat}, URL: ${downloadUrl}`);

    // Open download in new tab/window
    window.open(downloadUrl, '_blank');
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);
