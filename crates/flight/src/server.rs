// SPDX-License-Identifier: Apache-2.0

//! The Flight service (SPEC §8): `GetFlightInfo(ReadRequest)` → schema +
//! manifest + endpoints; `DoGet(ticket)` → dictionary batches then
//! RecordBatches, segment order, docid order. Effective config is echoed
//! in `app_metadata` (§8.3) — clients must not rely on unechoed defaults.

use std::pin::Pin;

use arrow_flight::encode::{DictionaryHandling, FlightDataEncoderBuilder};
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::stream::{self, BoxStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::engine::{self, ResolvedRead};
use crate::request::ReadRequest;

#[derive(Default)]
pub struct LuceneFlightService {}

impl LuceneFlightService {
    pub fn into_server(self) -> FlightServiceServer<Self> {
        FlightServiceServer::new(self)
    }
}

fn bad(e: lucene_arrow_core::Error) -> Status {
    match e {
        lucene_arrow_core::Error::InvalidArgument(m) => Status::invalid_argument(m),
        lucene_arrow_core::Error::Unsupported(m) => Status::unimplemented(m),
        other => Status::internal(other.to_string()),
    }
}

fn parse_request(bytes: &[u8]) -> Result<ReadRequest, Status> {
    serde_json::from_slice(bytes)
        .map_err(|e| Status::invalid_argument(format!("bad ReadRequest JSON: {e}")))
}

#[tonic::async_trait]
impl FlightService for LuceneFlightService {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = Pin<Box<dyn futures::Stream<Item = Result<FlightData, Status>> + Send>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let read_request = parse_request(&descriptor.cmd)?;
        let resolved = engine::resolve(&read_request).map_err(bad)?;

        // Ticket = the original request (stateless server); one endpoint
        // for now — endpoint grouping to ~1 GiB (reg. #10) comes with the
        // multi-endpoint planner.
        let ticket = Ticket { ticket: descriptor.cmd.clone() };
        let endpoint = FlightEndpoint::new().with_ticket(ticket);

        let total_rows: i64 = resolved
            .segments
            .iter()
            .map(|&o| resolved.dir.segments()[o].max_doc as i64)
            .sum();
        let config = serde_json::to_vec(&resolved.config)
            .map_err(|e| Status::internal(e.to_string()))?;

        let info = FlightInfo::new()
            .try_with_schema(&resolved.schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_endpoint(endpoint)
            .with_descriptor(descriptor)
            .with_total_records(total_rows)
            .with_app_metadata(config);
        Ok(Response::new(info))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let read_request = parse_request(&ticket.ticket)?;
        // Decode runs blocking (mmap + CPU executors) — keep it off the
        // async reactor.
        let batches = tokio::task::spawn_blocking(move || -> Result<_, Status> {
            // SPEC §7.8: the postings relation is a separate, segment-scoped
            // stream selected via `relation: "postings"`.
            if read_request.relation.as_deref() == Some("postings") {
                let field = read_request
                    .postings_field
                    .as_deref()
                    .ok_or_else(|| Status::invalid_argument("postings_field required"))?;
                let dir = lucene_arrow_codec::SegmentDirectory::open(&read_request.path)
                    .map_err(bad)?;
                let batch_rows = read_request.batch_rows.unwrap_or(1 << 20) as usize;
                let mut out = Vec::new();
                for seg in dir.segments() {
                    out.extend(engine::postings_batches(&dir, seg, field, batch_rows).map_err(bad)?);
                }
                let schema = out
                    .first()
                    .map(|b| b.schema())
                    .ok_or_else(|| Status::not_found("no postings"))?;
                return Ok((schema, out));
            }
            let resolved: ResolvedRead = engine::resolve(&read_request).map_err(bad)?;
            let mut out = Vec::new();
            let mut doc_base = 0i64;
            for &ord in &resolved.segments {
                out.extend(engine::segment_batches(&resolved, ord, doc_base).map_err(bad)?);
                doc_base += resolved.dir.segments()[ord].max_doc as i64;
            }
            Ok((resolved.schema, out))
        })
        .await
        .map_err(|e| Status::internal(format!("decode task: {e}")))??;
        let (schema, batches) = batches;

        // dict=segment ⇒ dictionaries are replaced at segment boundaries
        // via IPC dictionary replacement (SPEC §7.3): Resend, not delta.
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .with_dictionary_handling(DictionaryHandling::Resend)
            .build(stream::iter(batches.into_iter().map(Ok)))
            .map(|r| r.map_err(|e| Status::internal(e.to_string())));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn handshake(
        &self,
        _: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Ok(Response::new(stream::empty().boxed()))
    }
    async fn list_flights(
        &self,
        _: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }
    async fn poll_flight_info(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }
    async fn get_schema(
        &self,
        _: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        use futures::TryStreamExt;
        // First message carries the WriteJob (descriptor.cmd) + schema;
        // peel it off, parse the job, then chain it back for decoding.
        let mut raw = request.into_inner();
        let first = raw
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("empty put stream"))?;
        let descriptor = first
            .flight_descriptor
            .clone()
            .ok_or_else(|| Status::invalid_argument("first message needs a WriteJob descriptor"))?;
        let job: crate::request::WriteJob = serde_json::from_slice(&descriptor.cmd)
            .map_err(|e| Status::invalid_argument(format!("bad WriteJob JSON: {e}")))?;

        let rest = futures::stream::unfold(raw, |mut raw| async {
            match raw.message().await {
                Ok(Some(d)) => Some((Ok(d), raw)),
                Ok(None) => None,
                Err(e) => Some((Err(arrow_flight::error::FlightError::Tonic(Box::new(e))), raw)),
            }
        });
        let all = stream::iter([Ok(first)]).chain(rest);
        let mut decoder =
            arrow_flight::decode::FlightRecordBatchStream::new_from_flight_data(all);

        let mut session: Option<crate::write::WriteSession> = None;
        while let Some(batch) = decoder
            .try_next()
            .await
            .map_err(|e| Status::invalid_argument(format!("put stream: {e}")))?
        {
            if session.is_none() {
                session = Some(
                    crate::write::WriteSession::new(&job, &batch.schema()).map_err(bad)?,
                );
            }
            let s = session.as_mut().expect("initialized above");
            tokio::task::block_in_place(|| s.push(&batch)).map_err(bad)?;
        }
        let session = session
            .ok_or_else(|| Status::invalid_argument("put stream carried no record batches"))?;
        let manifests =
            tokio::task::block_in_place(move || session.finish()).map_err(bad)?;

        let results: Vec<Result<PutResult, Status>> = manifests
            .into_iter()
            .map(|m| {
                Ok(PutResult {
                    app_metadata: serde_json::to_vec(&m)
                        .map_err(|e| Status::internal(e.to_string()))?
                        .into(),
                })
            })
            .collect();
        Ok(Response::new(stream::iter(results).boxed()))
    }
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        match action.r#type.as_str() {
            "hydrate" => {
                let req: crate::request::HydrateRequest = serde_json::from_slice(&action.body)
                    .map_err(|e| Status::invalid_argument(format!("bad HydrateRequest: {e}")))?;
                let ipc = tokio::task::spawn_blocking(move || crate::hydrate::hydrate_ipc(&req))
                    .await
                    .map_err(|e| Status::internal(format!("hydrate task: {e}")))?
                    .map_err(bad)?;
                let result = arrow_flight::Result { body: ipc.into() };
                Ok(Response::new(stream::iter([Ok(result)]).boxed()))
            }
            "stats" => {
                #[derive(serde::Deserialize)]
                struct StatsRequest {
                    path: String,
                }
                let req: StatsRequest = serde_json::from_slice(&action.body)
                    .map_err(|e| Status::invalid_argument(format!("bad stats request: {e}")))?;
                let value = tokio::task::spawn_blocking(move || crate::engine::stats(&req.path))
                    .await
                    .map_err(|e| Status::internal(format!("stats task: {e}")))?
                    .map_err(bad)?;
                let body = serde_json::to_vec(&value).map_err(|e| Status::internal(e.to_string()))?;
                Ok(Response::new(stream::iter([Ok(arrow_flight::Result { body: body.into() })]).boxed()))
            }
            other => Err(Status::unimplemented(format!("unknown action {other:?}"))),
        }
    }
    async fn list_actions(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(
            stream::iter([
                Ok(ActionType {
                    r#type: "hydrate".into(),
                    description: "stored columns for (_seg,_doc) pairs (SPEC §7.4)".into(),
                }),
                Ok(ActionType {
                    r#type: "stats".into(),
                    description: "per-field decode-cost estimates (SPEC §8.1)".into(),
                }),
            ])
            .boxed(),
        ))
    }
    async fn do_exchange(
        &self,
        _: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }
}
