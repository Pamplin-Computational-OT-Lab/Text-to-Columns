library(shiny)
library(reticulate)
library(DT)
library(dplyr)
library(stringr)
library(pdftools)
library(tidyr)
library(future)
library(furrr)  # For parallel processing

# ------------------------------------------------------------------------------
# 1) Python Integration
# ------------------------------------------------------------------------------
source_python("zotero_integration.py")
source_python("classify_text.py")  # Python script for text classification

####################### Install Python and Required Packages #########################################

# Search for existing Python installations
find_python <- function() {
  python_paths <- c(
    Sys.which("python"),
    Sys.which("python3"),
    Sys.getenv("PYTHON"),
    Sys.getenv("PYTHON3")
  )
  python_paths <- unique(python_paths[python_paths != ""]) # Remove empty paths
  return(python_paths)
}

python_paths <- find_python()

if (length(python_paths) > 0) {
  # Use the first available Python installation
  message("Using existing Python installation at: ", python_paths[1])
  reticulate::use_python(python_paths[1], required = TRUE)
} else {
  # If no Python is found, install Miniconda
  if (!"r-miniconda" %in% reticulate::conda_list()$name) {
    message("No Python installation found. Installing Miniconda...")
    reticulate::install_miniconda()
    reticulate::use_condaenv("r-miniconda", required = TRUE)
    message("Miniconda installed and set up.")
  } else {
    reticulate::use_condaenv("r-miniconda", required = TRUE)
  }
}

# Check and install missing Python packages
required_packages <- c("requests", "beautifulsoup4", "numpy","transformers","torch","pandas","openai")
for (pkg in required_packages) {
  if (!reticulate::py_module_available(pkg)) {
    message(paste("Installing Python package:", pkg))
    reticulate::py_install(pkg)
  } else {
    message(paste("Python package already available:", pkg))
  }
}

# ------------------------------------------------------------------------------
# 2) PDF Context Extraction Function
# ------------------------------------------------------------------------------
extract_data_from_pdf <- function(pdf_path, search_terms, n) {
  pages <- pdf_text(pdf_path)
  content_full <- tolower(paste(pages, collapse = " "))
  
  doc_title <- basename(pdf_path)
  content_words <- unlist(str_split(content_full, "\\s+"))
  
  contexts_list <- lapply(search_terms, function(term) {
    term_lc <- tolower(term)
    term_indices <- which(content_words == term_lc)
    
    lapply(term_indices, function(idx) {
      start <- max(1, idx - n)
      end   <- min(length(content_words), idx + n)
      snippet <- paste(content_words[start:end], collapse = " ")
      snippet
    })
  })
  
  if (length(unlist(contexts_list)) == 0) {
    return(data.frame(
      Document     = character(0),
      Matched_Word = character(0),
      Context      = character(0),
      stringsAsFactors = FALSE
    ))
  }
  
  data.frame(
    Document     = doc_title,
    Matched_Word = rep(search_terms, lengths(contexts_list)),
    Context      = unlist(contexts_list),
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------------------------
# 3) Shiny UI
# ------------------------------------------------------------------------------
ui <- fluidPage(
  titlePanel("Pamplin COT Lab: Text-To-Columns"),
  
  sidebarLayout(
    sidebarPanel(
      textInput("zotero_db", 
                "Path to zotero.sqlite",
                value = "C:/Users/14074/Zotero/zotero.sqlite"),
      textInput("zotero_dir", 
                "Path to Zotero 'storage' Directory",
                value = "C:/Users/14074/Zotero/storage"),
      
      actionButton("load_collections", "Load Collections"),
      selectInput("collection_name", "Select a Collection", 
                  choices = c(), selected = NULL),
      actionButton("show_items", "Show Items in Collection"),
      br(), br(),
      
      textInput("search_terms", 
                "Search Terms (comma-separated)",
                value = "differentiation, competition, competitor, strategy"),
      numericInput("context_n", 
                   "Context Words (n)", value = 50, min = 1, step = 1),
      actionButton("process_pdfs", "Process PDFs"),
      br(),
      textOutput("status_text"),
      textOutput("process_status")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Items in Collection", 
                 DTOutput("items_table")),
        tabPanel("Search Results", 
                 DTOutput("snippets_table")),
        tabPanel("Text Classification",
                 fluidPage(
                   titlePanel("Text Classification"),
                   sidebarLayout(
                     sidebarPanel(
                       selectInput("text_column", 
                                   "Select Text Column", 
                                   choices = c(),  # Populated dynamically
                                   selected = NULL),
                       textInput("new_column", 
                                 "Name of New Column", 
                                 value = "Classification"),
                       textAreaInput("classification_prompt", 
                                     "Classification Prompt", 
                                     "Please classify the following text based on the terms provided."),
                       textInput("classification_terms", 
                                 "Classification Terms (comma-separated)", 
                                 value = "positive, negative, neutral"),
                       actionButton("run_classification", "Run Classification"),
                       textOutput("classification_status")
                     ),
                     mainPanel(
                       tableOutput("classification_preview")
                     )
                   )
                 )),
        tabPanel("Q&A with GPT (Conversation)",
                 fluidPage(
                   titlePanel("Q&A with GPT (Conversation Mode)"),
                   sidebarLayout(
                     sidebarPanel(
                       textInput("qa_question", "Enter Your Question", value = "What insights can I derive from the dataset?"),
                       selectInput("qa_columns", "Select Columns for Context", choices = NULL, multiple = TRUE),
                       numericInput("qa_sample_size", "Number of Rows to Include", value = 5, min = 1),
                       actionButton("run_qa", "Send Message"),
                       actionButton("reset_conversation", "Reset Conversation"),
                       textOutput("qa_status")
                     ),
                     mainPanel(
                       verbatimTextOutput("conversation_history")
                     )
                   )
                 )),
        
        tabPanel("Instructions",
                 p("1) Load collections and select a collection to view items."),
                 p("2) Use 'Process PDFs' to search PDFs in matched subfolders for specified terms."),
                 p("3) Use 'Text Classification' to classify the extracted snippets."),
                 p("4) Results will be displayed in respective tabs.")
        )
      )
    )
  )
)

# ------------------------------------------------------------------------------
# 4) Shiny Server
# ------------------------------------------------------------------------------
server <- function(input, output, session) {
  
  # Reactives for storing data
  collections_list <- reactiveVal(NULL)
  items_metadata   <- reactiveVal(NULL)
  data_store       <- reactiveVal(NULL)
  
  # ---------------------------------------------------------------------------
  # Load Collections
  # ---------------------------------------------------------------------------
  observeEvent(input$load_collections, {
    req(input$zotero_db)
    coll_names <- NULL
    tryCatch({
      coll_names <- get_all_collections(db_path = input$zotero_db)
    }, error = function(e) {
      showNotification(paste("Error loading collections:", e$message), type = "error")
    })
    collections_list(coll_names)
    updateSelectInput(session, "collection_name", choices = coll_names)
  })
  
  
  # ---------------------------------------------------------------------------
  # Update Select Column Dropdown
  # ---------------------------------------------------------------------------
  # Dynamically update 'Select Text Column' dropdown after processing PDFs
  observe({
    req(data_store())
    tryCatch({
      column_names <- names(data_store())
      updateSelectInput(session, "text_column", choices = column_names, selected = column_names[1])
    }, error = function(e) {
      message("Error updating text column dropdown: ", e$message)
      updateSelectInput(session, "text_column", choices = NULL)
    })
  })
  

  
  
  # ---------------------------------------------------------------------------
  # Show Items in Collection
  # ---------------------------------------------------------------------------
  observeEvent(input$show_items, {
    req(input$zotero_db, input$collection_name)
    meta_list <- NULL
    tryCatch({
      meta_list <- get_collection_items_metadata(
        db_path = input$zotero_db,
        collection_name = input$collection_name,
        require_attachment = FALSE
      )
    }, error = function(e) {
      showNotification(paste("Error fetching item metadata:", e$message), type = "error")
    })
    if (!is.null(meta_list) && length(meta_list) > 0) {
      df_list <- lapply(meta_list, function(x) {
        x[["title"]] <- ifelse(is.null(x[["title"]]) || x[["title"]] == "", "(No Title)", x[["title"]])
        x[["authors"]] <- ifelse(is.null(x[["authors"]]) || x[["authors"]] == "", "(No Authors)", x[["authors"]])
        x[["year"]] <- ifelse(is.null(x[["year"]]) || x[["year"]] == "", "(No Year)", x[["year"]])
        x[["key"]] <- ifelse(is.null(x[["key"]]) || x[["key"]] == "", "(No Folder)", x[["key"]])
        as.data.frame(x, stringsAsFactors = FALSE)
      })
      df <- dplyr::bind_rows(df_list)
      items_metadata(df)
    } else {
      items_metadata(NULL)
    }
  })
  
  output$items_table <- renderDT({
    df <- items_metadata()
    req(df)
    datatable(df, rownames = FALSE, options = list(pageLength = 10, autoWidth = TRUE),
              colnames = c("Item ID", "Title", "Authors", "Year", "Folder Name"))
  })
  
  # ---------------------------------------------------------------------------
  # Process PDFs
  # ---------------------------------------------------------------------------
  observeEvent(input$process_pdfs, {
    req(items_metadata(), input$zotero_dir, input$search_terms)
    df <- items_metadata()
    valid_folders <- file.path(input$zotero_dir, df$key)
    pdf_files <- unlist(lapply(valid_folders, function(folder) {
      list.files(folder, pattern = "\\.pdf$", full.names = TRUE)
    }))
    search_terms <- str_split(input$search_terms, ",\\s*")[[1]]
    withProgress(message="Processing PDFs", value=0, {
      pdf_data_list <- lapply(seq_along(pdf_files), function(i) {
        incProgress(1/length(pdf_files))
        extract_data_from_pdf(pdf_files[i], search_terms, input$context_n)
      })
      pdf_data <- bind_rows(pdf_data_list)
      existing <- data_store()
      if (!is.null(existing)) {
        pdf_data <- bind_rows(existing, pdf_data)
      }
      data_store(pdf_data)
    })
    output$process_status <- renderText({
      paste("Processing completed! Found", nrow(data_store()), "snippets total.")
    })
  })
  
  output$snippets_table <- renderDT({
    req(data_store())
    datatable(data_store(), rownames = FALSE, options = list(pageLength = 10, autoWidth = TRUE),
              colnames = c("Document", "Matched Word", "Context"))
  })
  
  
  # ---------------------------------------------------------------------------
  # Run Text Classification
  # ---------------------------------------------------------------------------
  observeEvent(input$run_classification, {
    req(data_store(), input$text_column, input$new_column, input$classification_prompt, input$classification_terms)
    
    # Prepare inputs
    dataset <- data_store()
    text_column <- dataset[[input$text_column]]
    prompt <- input$classification_prompt
    terms <- str_split(input$classification_terms, ",\\s*")[[1]]
    
    # Run classification
    withProgress(message = "Running Classification", value = 0, {
      results <- future_map(text_column, ~ classify_text(list(.x), prompt, terms), .progress = TRUE)
      dataset[[input$new_column]] <- unlist(results)
      data_store(dataset)
    })
    
    output$classification_status <- renderText("Classification completed successfully!")
    output$classification_preview <- renderTable(head(dataset))
  })
  
  # ------------------------------------------------------------------------------
  # Q&A with Conversation History
  # ------------------------------------------------------------------------------
  # Q&A with Conversation History
  # ------------------------------------------------------------------------------
  
  # Initialize the conversation history
  initialize_conversation <- function() {
    list(list(role = "researcher", content = "Hi assistant. Good to see you agin. Let's get started with todays research task."))
  }
  
  conversation_history <- reactiveVal(initialize_conversation())
  
  # Update the columns dropdown for context
  observe({
    req(data_store())
    tryCatch({
      colnames <- names(data_store())
      updateSelectInput(session, "qa_columns", choices = colnames, selected = colnames)
    }, error = function(e) {
      message("Error updating columns for context: ", e$message)
      updateSelectInput(session, "qa_columns", choices = NULL)
    })
  })
  
  # Handle Q&A execution
  process_qa <- function(conversation, question, context_string) {
    # Add the user's input to the conversation
    conversation <- append(conversation, list(
      list(role = "researcher", content = paste("Use this data as your primary source:\n", context_string, "\n\n", question))
    ))
    
    # Call the Python function with the updated conversation
    updated_conversation <- qa_with_gpt2(conversation)  # Updated function with chunk handling
    return(updated_conversation)
  }
  
  
  observeEvent(input$run_qa, {
    req(data_store(), input$qa_question, input$qa_columns, input$qa_sample_size)
    
    # Subset dataset for selected columns and rows
    dataset <- data_store()
    selected_columns <- input$qa_columns
    context_data <- dataset %>%
      select(all_of(selected_columns)) %>%
      sample_n(min(nrow(dataset), input$qa_sample_size))
    
    # Construct context string
    context_string <- paste(
      apply(context_data, 1, function(row) paste(names(row), row, sep = ": ", collapse = "; ")),
      collapse = "\n"
    )
    
    # Append user question to conversation history
    conversation <- conversation_history()
    conversation <- append(conversation, list(
      list(role = "researcher", content = paste("Data:\n", context_string, "\n\n", input$qa_question))
    ))
    
    # Call Python for Q&A
    withProgress(message = "Processing Q&A", value = 0, {
      tryCatch({
        conversation <- qa_with_gpt2(conversation)  # Updated Python function
        conversation_history(conversation)  # Save updated history
        output$qa_status <- renderText("Message sent successfully!")
      }, error = function(e) {
        output$qa_status <- renderText(paste("Error during Q&A:", e$message))
      })
    })
  })
  
  
  
  
  # Display the conversation history
  output$conversation_history <- renderText({
    conversation <- conversation_history()
    paste(
      sapply(conversation, function(msg) paste0(msg$role, ": ", msg$content)),
      collapse = "\n\n"
    )
  })
  
  # Reset the conversation history
  observeEvent(input$reset_conversation, {
    conversation_history(initialize_conversation())
    output$qa_status <- renderText("Conversation reset successfully!")
  })
  
}

shinyApp(ui = ui, server = server)
